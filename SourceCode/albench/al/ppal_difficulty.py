"""PPAL DCUS difficulty (Yang et al., CVPR 2024, arXiv:2211.11612)."""
import math

import numpy as np


# IoU and matching

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # pairwise IoU between two box sets -> (len(a), len(b))
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_post_nms_to_gt(pred_boxes, pred_labels, pred_scores,
                         gt_boxes, gt_labels) -> list:
    # greedy match: each det -> an unused same-class GT with max IoU
    pred_labels = np.asarray(pred_labels)
    gt_labels = np.asarray(gt_labels)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []
    iou = iou_xyxy(np.asarray(pred_boxes, float), np.asarray(gt_boxes, float))
    order = np.argsort(-np.asarray(pred_scores, float))
    used = set()
    out = []
    for di in order:
        same = np.where(gt_labels == pred_labels[di])[0]
        same = [g for g in same if g not in used]
        if not same:
            continue
        best = max(same, key=lambda g: iou[di, g])
        if iou[di, best] <= 0.0:
            continue
        used.add(best)
        out.append((int(pred_labels[di]), float(pred_scores[di]),
                    float(iou[di, best])))
    return out


# DCUS quality EMA

def avg_q_from_matches(matches: list, nc: int, xi: float) -> np.ndarray:
    # per-class mean quality  q = score^xi * IoU^(1-xi)
    s = np.zeros(nc, float)
    c = np.zeros(nc, float)
    for label, score, iou in matches:
        s[label] += (score ** xi) * (iou ** (1.0 - xi))
        c[label] += 1.0
    return np.where(c > 0, s / (c + 1e-5), 0.0)


class QualityEMA:
    def __init__(self, nc: int, base_momentum: float):
        self.class_quality = np.zeros(nc, float)
        self.class_momentum = np.full(nc, base_momentum, float)
        self.base_momentum = base_momentum

    def update(self, avg_q: np.ndarray) -> None:
        # class seen this batch -> momentum resets to base; unseen -> momentum decays further
        m = self.class_momentum
        self.class_quality = m * self.class_quality + (1.0 - m) * avg_q
        self.class_momentum = np.where(
            avg_q > 0, self.base_momentum,
            self.class_momentum * self.base_momentum)

    @property
    def value(self) -> np.ndarray:
        return self.class_quality.copy()


def _xywhn_to_xyxyn(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, float)
    if len(b) == 0:
        return np.zeros((0, 4), float)
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


# Ultralytics training callback

def make_batch_quality_callbacks(state: "QualityEMA", xi: float,
                                 score_thr: float, *, nms_iou: float = 0.45,
                                 max_det: int = 300) -> dict:
    """Per-batch DCUS difficulty EMA callbacks (Ultralytics {event: fn} dict)."""
    nc = state.class_quality.shape[0]

    def _on_train_start(trainer) -> None:
        orig = trainer.preprocess_batch

        def _wrapped(batch):
            b = orig(batch)
            trainer._ppal_batch = b          # stash batch for _on_train_batch_end
            return b

        trainer.preprocess_batch = _wrapped

    def _on_train_batch_end(trainer) -> None:
        import torch
        from ultralytics.utils.nms import non_max_suppression
        batch = getattr(trainer, "_ppal_batch", None)
        if batch is None:
            return
        model = trainer.model
        img = batch["img"]
        was_training = model.training
        # run inference to compute difficulty, then restore training mode
        model.eval()
        try:
            with torch.no_grad():
                out = model(img)
            y = out[0] if isinstance(out, (tuple, list)) else out
            dets = non_max_suppression(y, conf_thres=score_thr,
                                       iou_thres=nms_iou, nc=nc,
                                       max_det=max_det)
        finally:
            if was_training:
                model.train()
        H = float(img.shape[-2])
        W = float(img.shape[-1])

        def _np(x):
            return (x.detach().cpu().numpy() if hasattr(x, "detach")
                    else np.asarray(x))
        bi = _np(batch["batch_idx"]).reshape(-1).astype(int)
        gcls = _np(batch["cls"]).reshape(-1).astype(int)
        gbox = _np(batch["bboxes"]).astype(float)            # xywh norm
        matches = []
        for i, d in enumerate(dets):
            d = d.detach().cpu().numpy()
            if len(d):
                pb = d[:, :4].astype(float)
                pb[:, [0, 2]] /= W
                pb[:, [1, 3]] /= H
                ps = d[:, 4].astype(float)
                pl = d[:, 5].astype(int)
            else:
                pb, ps, pl = np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
            m = bi == i
            matches += match_post_nms_to_gt(
                pb, pl, ps, _xywhn_to_xyxyn(gbox[m]), gcls[m])
        state.update(avg_q_from_matches(matches, nc, xi))

    return {"on_train_start": _on_train_start,
            "on_train_batch_end": _on_train_batch_end}


# Class reweighting and I/O

def reweight(class_quality: np.ndarray, alpha: float, ub: float) -> np.ndarray:
    # DCUS Eq.4: low-quality class -> higher weight
    d = 1.0 - np.asarray(class_quality, float)
    b = math.exp(1.0 / alpha) - 1.0
    return 1.0 + alpha * np.log(b * d + 1.0) * ub


def save_quality(path: str, arr: np.ndarray) -> None:
    np.save(path, np.asarray(arr, float))


def load_quality(path: str) -> np.ndarray:
    return np.load(path).astype(float)
