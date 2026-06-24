"""PPAL CCMS per-detection FPN features (Yang et al., CVPR 2024,
arXiv:2211.11612). Inference-only."""
import numpy as np


# Coordinate helpers

def anchor_level(idx: int, splits: tuple) -> int:
    # FPN level holding anchor `idx` (splits = #anchors per level, concat order)
    acc = 0
    for lvl, n in enumerate(splits):
        acc += n
        if idx < acc:
            return lvl
    raise ValueError(f"anchor idx {idx} out of range for splits {splits}")


def box_center_grid_coords(box: np.ndarray, img_w: float, img_h: float):
    # box centre -> grid_sample coords in [-1, 1]
    cx = ((0.5 * (box[0] + box[2]) / img_w) - 0.5) * 2.0
    cy = ((0.5 * (box[1] + box[3]) / img_h) - 0.5) * 2.0
    return float(cx), float(cy)


def letterbox_xyxy(xyxy: np.ndarray, orig_h: float, orig_w: float,
                   imgsz: int) -> np.ndarray:
    # boxes ORIG -> letterboxed coords: ratio = imgsz/max(H,W), symmetric pad
    ratio = imgsz / float(max(orig_h, orig_w))
    h_scaled = orig_h * ratio
    w_scaled = orig_w * ratio
    pad_h = (imgsz - h_scaled) / 2.0
    pad_w = (imgsz - w_scaled) / 2.0
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    return xyxy * ratio + np.array([pad_w, pad_h, pad_w, pad_h])


# Ultralytics model patching
import contextlib
from types import MethodType


@contextlib.contextmanager
def _patch_predict_once(net, layers):
    # rebind net._predict_once to capture FPN feature maps; restore on exit
    had_instance_attr = "_predict_once" in net.__dict__
    saved = net.__dict__.get("_predict_once")
    net._predict_once = MethodType(_predict_once_factory(layers), net)
    try:
        yield
    finally:
        if had_instance_attr:
            net._predict_once = saved
        else:
            del net._predict_once   # unshadow → class descriptor resumes
        if "_ppal_feats" in net.__dict__:
            delattr(net, "_ppal_feats")


def _build_feature_predictor():
    """DetectionPredictor subclass that attaches per-kept-box anchor indices."""
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms

    class FeaturePredictor(DetectionPredictor):
        def postprocess(self, preds, img, orig_imgs, **kwargs):
            raw = preds[0] if isinstance(preds, (list, tuple)) else preds
            out, keepi = nms.non_max_suppression(
                raw, self.args.conf, self.args.iou, self.args.classes,
                self.args.agnostic_nms, max_det=self.args.max_det,
                nc=0, return_idxs=True)
            scores = raw.transpose(-1, -2)[..., 4:]
            results = self.construct_results(out, img, orig_imgs)
            for r, kept, sc in zip(results, keepi, scores):
                r.cls_scores = sc[kept.long()] if kept.numel() else sc[:0]
                r._keep_idxs = kept
            return results

    return FeaturePredictor


def _predict_once_factory(embed_layers):
    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, feats = [], []
        for m in self.model:
            if m.f != -1:
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i in embed_layers:
                feats.append(x)
            if m.i == max(embed_layers):
                self._ppal_feats = feats
        return x
    return _predict_once


# Feature extraction

def extract_det_features(model, paths: list, cfg: dict):
    # grid_sample the box centre on its FPN level -> per-detection feature vector
    import torch
    import torch.nn.functional as F
    from .device import device

    pp = cfg["al"]["ppal"]
    layers = list(pp["embed_layers"])
    imgsz = cfg["al"]["imgsz"]
    net = model.model
    FeaturePredictor = _build_feature_predictor()

    feats_out, labels_out, scores_out = [], [], []
    max_ch = None
    # reset so predict() installs FeaturePredictor
    model.predictor = None
    try:
        with _patch_predict_once(net, layers):
            for path in paths:
                # rect=False => symmetric square pad
                res = model.predict(path, predictor=FeaturePredictor,
                                    conf=pp["score_thr"], imgsz=imgsz,
                                    rect=False, device=device(),
                                    verbose=False)[0]
                keepi = getattr(res, "_keep_idxs", None)
                if keepi is None:
                    raise RuntimeError(
                        "ppal: _keep_idxs missing, FeaturePredictor was not "
                        "used (YOLO.predict reused a stale predictor), "
                        "Stage-2 features would be silently empty")
                lvl_feats = [f[0] for f in net._ppal_feats]    # C,H,W per lvl
                splits = tuple(int(f.shape[1] * f.shape[2]) for f in lvl_feats)
                # neck channels differ per level, zero-pad to max(C) for cosine
                if max_ch is None:
                    max_ch = max(int(f.shape[0]) for f in lvl_feats)
                raw = res.boxes
                xyxy = raw.xyxy.detach().cpu().numpy()
                cls = raw.cls.detach().cpu().numpy().astype(int)
                conf = raw.conf.detach().cpu().numpy()
                # xyxy is in ORIG coords, map to the letterboxed frame
                H_orig, W_orig = res.orig_shape
                if len(xyxy) == 0:
                    feats_out.append(np.zeros((0, max_ch)))
                    labels_out.append(np.zeros((0,), int))
                    scores_out.append(np.zeros((0,), float))
                    continue
                xyxy_lb = letterbox_xyxy(xyxy, H_orig, W_orig, imgsz)
                vecs = []
                for di in range(len(xyxy)):
                    lvl = anchor_level(int(keepi[di]), splits)
                    cx, cy = box_center_grid_coords(xyxy_lb[di], imgsz, imgsz)
                    grid = torch.tensor([[[[cx, cy]]]],
                                        dtype=lvl_feats[lvl].dtype,
                                        device=lvl_feats[lvl].device)
                    v = F.grid_sample(lvl_feats[lvl][None], grid,
                                      mode="bilinear", align_corners=False)
                    v = v.reshape(-1).detach().cpu().numpy()
                    if v.shape[0] < max_ch:        # zero-pad to max(C)
                        v = np.concatenate([v, np.zeros(max_ch - v.shape[0])])
                    vecs.append(v)
                feats_out.append(np.vstack(vecs))
                labels_out.append(cls)
                scores_out.append(conf)
    finally:
        model.predictor = None
    return feats_out, labels_out, scores_out
