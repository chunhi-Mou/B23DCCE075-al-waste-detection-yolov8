"""Gradio UI for the YOLOv8 waste-detection demo (PTIT 2026).

    python -m demo.app              # local + temporary public share link
    python -m demo.app --no-share   # local only
Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from demo import charts, content, engine
from demo.icons import svg

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

THEME = gr.themes.Soft(
    primary_hue="green", secondary_hue="emerald", neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
    spacing_size="sm", radius_size="sm", text_size="sm",
)

CSS = """
.gradio-container {max-width: 1180px !important; width: 100% !important; margin: 0 auto !important; padding-top: 4px !important;}
footer {display: none !important;}
#app-header {padding: 2px 2px 4px;}
.tab-hint {color:#64748b; font-size:12.5px; margin:-2px 0 4px;}
.result-panel {max-height: 250px; overflow-y: auto; padding-right: 4px;}
"""

FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

STOP_CAMERA_JS = """
() => {
  document.querySelectorAll('video').forEach((v) => {
    const s = v.srcObject;
    if (s && s.getTracks) {
      s.getTracks().forEach((t) => t.stop());
      v.srcObject = null;
    }
  });
}
"""

_HEADER_HTML = f"""
<div id="app-header" style="display:flex;align-items:center;gap:14px">
  {svg("recycle", 34, "#16a34a")}
  <div>
    <div style="font-size:1.35rem;font-weight:700;line-height:1.2">Phân loại rác thải tái chế</div>
    <div style="color:#6b7280;font-size:0.84rem;margin-top:2px">
      YOLOv8n · Active Learning · PTIT 2026 · 5 lớp: bìa carton, giấy, thuỷ tinh, kim loại, nhựa
    </div>
  </div>
</div>
"""


def _hint(icon: str, text: str) -> str:
    return (f'<div class="tab-hint" style="display:flex;align-items:center;gap:7px">'
            f'{svg(icon, 15, "#16a34a")}<span>{text}</span></div>')


# Logic glue (UI <-> engine)

_CURATED: dict[str, engine.ModelInfo] | None = None
_ALL: dict[str, engine.ModelInfo] | None = None


def curated_models() -> dict[str, engine.ModelInfo]:
    global _CURATED
    if _CURATED is None:
        _CURATED = engine.discover_models()
    return _CURATED


def all_models() -> dict[str, engine.ModelInfo]:
    global _ALL
    if _ALL is None:
        _ALL = engine.discover_all_models()
    return _ALL


def _default_label(models: dict[str, engine.ModelInfo]) -> str | None:
    for label, info in models.items():
        if info.kind == "baseline":
            return label
    return next(iter(models), None)


def _example_paths() -> list[str]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return [str(p) for p in sorted(EXAMPLES_DIR.iterdir())
            if p.suffix.lower() in _IMG_EXTS]


def _infer(image, info: engine.ModelInfo, conf: float, iou: float):
    return engine.predict(engine.load_model(info.path), image, conf, iou)


def analyze_user(image, model_label):
    if image is None:
        return None, content.warn_html("Vui lòng tải lên hoặc chụp một ảnh.")
    models = curated_models()
    if not models:
        return None, content.err_html("Không tìm thấy model nào trong thư mục results/.")
    info = models.get(model_label) or next(iter(models.values()))
    try:
        res = _infer(image, info, DEFAULT_CONF, DEFAULT_IOU)
    except Exception as exc:  # noqa: BLE001
        return None, content.err_html(f"Lỗi khi phân tích: {exc}")
    return res.annotated, content.format_user_result(res.counts, res.detections)


def on_user_model_change(model_label):
    return content.model_info_md(curated_models().get(model_label))


def analyze_dev(image, model_label, conf, iou):
    if image is None:
        return None, [], [], "Vui lòng tải lên hoặc chụp một ảnh."
    info = all_models().get(model_label)
    if info is None:
        return None, [], [], "Không tìm thấy model đã chọn."
    try:
        res = _infer(image, info, conf, iou)
    except Exception as exc:  # noqa: BLE001
        return None, [], [], f"Lỗi: {exc}"
    raw = [[d["cls_name"], round(d["conf"], 3), *d["xyxy"]] for d in res.detections]
    counts = [[k, v] for k, v in sorted(res.counts.items())]
    return res.annotated, raw, counts, f"{res.ms:.1f} ms"


# Layout

def _user_tab() -> None:
    curated = curated_models()
    default = _default_label(curated)
    gr.HTML(_hint("search", "Tải ảnh hoặc chụp webcam, chọn mô hình rồi bấm Phân tích."))
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            img_in = gr.Image(sources=["upload", "webcam"], type="numpy",
                              label="Ảnh đầu vào", height=300)
            with gr.Row():
                model = gr.Dropdown(choices=list(curated), value=default, scale=3,
                                    label="Mô hình", interactive=bool(curated))
                btn = gr.Button("Phân tích", variant="primary", scale=1)
            info = gr.HTML(content.model_info_md(curated.get(default)))
            examples = _example_paths()
            if examples:
                gallery = gr.Gallery(value=examples, columns=6, height=120,
                                     allow_preview=False, label="Ảnh mẫu")

                def pick_example(evt: gr.SelectData):
                    return examples[evt.index]

                gallery.select(pick_example, None, img_in)
        with gr.Column(scale=1):
            img_out = gr.Image(label="Kết quả", type="numpy", height=300)
            result = gr.HTML(elem_classes="result-panel")
    model.change(on_user_model_change, inputs=model, outputs=info)
    btn.click(analyze_user, inputs=[img_in, model], outputs=[img_out, result])


def _results_tab() -> None:
    if not engine.al_results():
        gr.HTML(content.warn_html("Chưa có kết quả benchmark trong results/."))
        return
    gr.HTML(_hint("chart", "So sánh 4 chiến lược Active Learning trên cùng benchmark "
                           "(YOLOv8n, test mAP@50, 3 seed)."))
    with gr.Row(equal_height=False):
        gr.Image(value=charts.render(charts.comparison_curve()),
                 label="Hiệu năng theo % dữ liệu", interactive=False, height=320)
        gr.Image(value=charts.render(charts.per_class_bars()),
                 label="AP@50 theo lớp", interactive=False, height=320)
    with gr.Row(equal_height=False):
        gr.HTML(content.benchmark_table_html(engine.aubc_table(),
                                             engine.baseline_summary()))
        gr.HTML(content.results_note_html())


def _dev_tab() -> None:
    full = all_models()
    default = _default_label(full)
    gr.HTML(_hint("sliders", "Chọn bất kỳ model (mọi seed & vòng AL), chỉnh ngưỡng, "
                             "xem chi tiết phát hiện và thời gian inference."))
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            img_in = gr.Image(sources=["upload"], type="numpy",
                              label="Ảnh đầu vào", height=280)
            model = gr.Dropdown(choices=list(full), value=default,
                                label="Model", interactive=bool(full))
            with gr.Row():
                conf = gr.Slider(0.0, 1.0, value=DEFAULT_CONF, step=0.01, label="Confidence")
                iou = gr.Slider(0.0, 1.0, value=DEFAULT_IOU, step=0.01, label="IoU (NMS)")
            btn = gr.Button("Phân tích", variant="primary")
        with gr.Column(scale=1):
            img_out = gr.Image(label="Kết quả", type="numpy", height=280)
            latency = gr.Textbox(label="Thời gian inference")
            with gr.Row():
                raw = gr.Dataframe(headers=["class", "conf", "x1", "y1", "x2", "y2"],
                                   label="Chi tiết phát hiện")
                counts = gr.Dataframe(headers=["class", "count"], label="Theo lớp")
    btn.click(analyze_dev, inputs=[img_in, model, conf, iou],
              outputs=[img_out, raw, counts, latency])


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Phân loại rác thải · YOLOv8") as demo:
        gr.HTML(_HEADER_HTML)
        with gr.Tabs() as tabs:
            with gr.Tab("Phân tích"):
                _user_tab()
            with gr.Tab("Kết quả huấn luyện"):
                _results_tab()
            with gr.Tab("Dev / So sánh"):
                _dev_tab()
        tabs.select(None, js=STOP_CAMERA_JS)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 waste-detection Gradio demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", dest="share", action="store_true",
                        help="create a temporary public link (default)")
    parser.add_argument("--no-share", dest="share", action="store_false",
                        help="local only, no public link")
    parser.add_argument("--no-browser", dest="inbrowser", action="store_false",
                        help="do not auto-open the browser")
    parser.set_defaults(share=True, inbrowser=True)
    args = parser.parse_args()

    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=args.inbrowser,
                      theme=THEME, css=CSS, js=FORCE_LIGHT_JS)


if __name__ == "__main__":
    main()
