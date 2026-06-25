"""Vietnamese result rendering and the shared visual tokens for the demo UI."""

from __future__ import annotations

from demo.icons import svg

INK = "#0f172a"
MUTED = "#64748b"
LINE = "#e2e8f0"
SOFT = "#f8fafc"
GREEN = "#16a34a"
FONT = "'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"
MONO = "ui-monospace,'SF Mono','Cascadia Code',Consolas,monospace"

CLASS_COLOR: dict[str, str] = {
    "cardboard": "#b45309",
    "paper":     "#2563eb",
    "glass":     "#0891b2",
    "metal":     "#475569",
    "plastic":   "#15803d",
}

CLASS_VI: dict[str, str] = {
    "cardboard": "Bìa carton",
    "paper":     "Giấy",
    "glass":     "Thuỷ tinh",
    "metal":     "Kim loại",
    "plastic":   "Nhựa",
}

CLASS_TIPS: dict[str, list[str]] = {
    "cardboard": ["Gỡ băng keo và ghim, làm phẳng cho gọn",
                  "Giữ khô, cắt bỏ phần dính dầu mỡ"],
    "paper":     ["Gom giấy khô sạch, bỏ ghim kẹp",
                  "Tránh giấy dính dầu mỡ hoặc phủ nilon"],
    "glass":     ["Rửa sạch, tháo nắp để gom riêng",
                  "Bọc kỹ mảnh vỡ, không lẫn gốm sứ"],
    "metal":     ["Rửa sạch và bóp dẹp lon, hộp",
                  "Bình xịt phải dùng hết, để rỗng mới bỏ"],
    "plastic":   ["Rửa sạch, để khô, bóp dẹp, tháo nắp",
                  "Tái chế tốt số 1, 2, 5. Tránh số 3, 6, 7"],
}

_STRATEGY_NAMES = {"random": "Random", "uncertainty": "Uncertainty",
                   "coreset": "CoreSet", "ppal": "PPAL"}


def _banner(icon: str, color: str, fg: str, bg: str, border: str, msg: str) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:10px;padding:12px 14px;'
        f'font-family:{FONT};color:{fg};background:{bg};border:1px solid {border};'
        f'border-radius:10px">{svg(icon, 18, color)}<span>{msg}</span></div>'
    )


def warn_html(msg: str) -> str:
    return _banner("warn", "#d97706", "#92400e", "#fffbeb", "#fde68a", msg)


def err_html(msg: str) -> str:
    return _banner("error", "#dc2626", "#991b1b", "#fef2f2", "#fecaca", msg)


def _pill(text: str, mono: bool = False) -> str:
    fam = MONO if mono else FONT
    return (
        f'<span style="display:inline-block;padding:2px 9px;border-radius:999px;'
        f'background:{SOFT};border:1px solid {LINE};color:{MUTED};font-size:12px;'
        f'font-family:{fam}">{text}</span>'
    )


def model_info_md(info) -> str:
    """One styled blurb for the selected model (duck-typed engine.ModelInfo)."""
    if info is None:
        return ""
    if getattr(info, "kind", None) == "baseline":
        title, sub = "Baseline (Oracle)", "Huấn luyện trên 100% dữ liệu gán nhãn"
    else:
        title = _STRATEGY_NAMES.get(getattr(info, "strategy", "") or "", "AL")
        bits = ["Active Learning"]
        if info.seed is not None:
            bits.append(f"seed {info.seed}")
        if info.round is not None:
            bits.append(f"vòng {info.round}")
        sub = ", ".join(bits)

    pills = []
    if info.data_pct is not None:
        pills.append(_pill(f"{round(info.data_pct)}% dữ liệu"))
    if info.map50 is not None:
        pills.append(_pill(f"test mAP@50 {info.map50:.3f}", mono=True))
    pill_row = (f'<div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:7px">'
                f'{"".join(pills)}</div>') if pills else ""
    return (
        f'<div style="font-family:{FONT};line-height:1.35">'
        f'<div style="font-weight:650;color:{INK}">{title}</div>'
        f'<div style="font-size:12.5px;color:{MUTED}">{sub}</div>'
        f'{pill_row}</div>'
    )


def format_user_result(counts: dict[str, int], detections: list[dict]) -> str:
    if not detections:
        return _banner("search", "#94a3b8", "#475569", SOFT, LINE,
                       "Chưa phát hiện vật thể nào. Thử ảnh rõ nét và đủ sáng hơn.")

    best_conf: dict[str, float] = {}
    for det in detections:
        name = det["cls_name"]
        best_conf[name] = max(best_conf.get(name, 0.0), det["conf"])

    header = (
        f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:10px">'
        f'<span style="font-size:22px;font-weight:700;color:{INK}">{len(detections)}</span>'
        f'<span style="color:{MUTED};font-size:13px">vật thể, {len(counts)} '
        f'nhóm vật liệu</span></div>'
    )
    cards = "".join(_class_card(n, counts[n], best_conf.get(n, 0.0))
                    for n in sorted(counts, key=lambda k: -counts[k]))
    return (f'<div style="font-family:{FONT};display:flex;flex-direction:column;'
            f'gap:8px">{header}{cards}</div>')


def _chip(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:999px;'
        f'background:{color}1a;color:{color};border:1px solid {color}40;'
        f'font-size:11px;font-weight:600">{text}</span>'
    )


def _tip_rows(tips: list[str]) -> str:
    if not tips:
        return ""
    rows = "".join(
        f'<div style="display:flex;gap:7px;align-items:flex-start;margin-top:5px">'
        f'{svg("check", 14, GREEN)}'
        f'<span style="color:{INK};font-size:12.5px;line-height:1.4">{t}</span></div>'
        for t in tips
    )
    return (f'<div style="margin-top:8px;padding-top:8px;'
            f'border-top:1px dashed {LINE}">{rows}</div>')


def _class_card(name: str, count: int, conf: float) -> str:
    color = CLASS_COLOR.get(name, MUTED)
    vi = CLASS_VI.get(name, name)
    return (
        f'<div style="display:flex;gap:12px;padding:12px 14px;background:#fff;'
        f'border:1px solid {LINE};border-left:3px solid {color};border-radius:10px">'
        f'<div style="flex-shrink:0;margin-top:1px">{svg(name, 20, color)}</div>'
        f'<div style="flex:1;min-width:0">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f'<span style="font-weight:650;color:{INK}">{vi}</span>'
        f'{_chip("Tái chế", GREEN)}'
        f'<span style="margin-left:auto;font-family:{MONO};font-size:13px;'
        f'font-weight:600;color:{color}">{round(conf * 100)}%</span></div>'
        f'<div style="font-size:12px;color:{MUTED};margin-top:2px">'
        f'{name} · số lượng: {count}</div>{_tip_rows(CLASS_TIPS.get(name, []))}</div></div>'
    )


# Training-results tab

def _mean_of(cell: str) -> float:
    try:
        return float(str(cell).split("±")[0])
    except (TypeError, ValueError):
        return -1.0


def benchmark_table_html(aubc_rows: list[dict], baseline: dict | None) -> str:
    """Styled AUBC / mAP-at-cap table; best value per column highlighted green."""
    rows = sorted(aubc_rows, key=lambda r: r.get("strategy", ""))
    if not rows:
        return warn_html("Chưa có bảng AUBC trong results/.")
    best_aubc = max((_mean_of(r.get("AUBC")) for r in rows), default=-1)
    best_cap = max((_mean_of(r.get("mAP_at_cap")) for r in rows), default=-1)

    def cell(val: str, is_best: bool) -> str:
        style = f"font-weight:700;color:{GREEN}" if is_best else f"color:{INK}"
        return (f'<td style="padding:7px 10px;font-family:{MONO};font-size:12.5px;'
                f'text-align:right;{style}">{val}</td>')

    body = ""
    for r in rows:
        body += (
            f'<tr style="border-top:1px solid {LINE}">'
            f'<td style="padding:7px 10px;color:{INK};font-weight:600">{r.get("strategy","")}</td>'
            f'{cell(r.get("AUBC",""), _mean_of(r.get("AUBC")) == best_aubc)}'
            f'{cell(r.get("mAP_at_cap",""), _mean_of(r.get("mAP_at_cap")) == best_cap)}'
            f'<td style="padding:7px 10px;text-align:right;color:{MUTED}">{r.get("n_seeds","")}</td>'
            f'</tr>'
        )
    if baseline:
        body += (
            f'<tr style="border-top:2px solid {LINE};background:{SOFT}">'
            f'<td style="padding:7px 10px;color:{INK};font-weight:600">Train 100% dữ liệu</td>'
            f'<td style="padding:7px 10px;text-align:right;color:{MUTED}">n/a</td>'
            f'<td style="padding:7px 10px;font-family:{MONO};font-size:12.5px;'
            f'text-align:right;color:{INK}">{baseline["test_mAP50"]:.3f}</td>'
            f'<td style="padding:7px 10px;text-align:right;color:{MUTED}">1</td></tr>'
        )
    head = (
        f'<tr style="color:{MUTED};font-size:12px;text-align:right">'
        f'<th style="padding:7px 10px;text-align:left">Chiến lược</th>'
        f'<th style="padding:7px 10px">AUBC</th>'
        f'<th style="padding:7px 10px">mAP@50 ở 20%</th>'
        f'<th style="padding:7px 10px">#seed</th></tr>'
    )
    return (
        f'<div style="font-family:{FONT}">'
        f'<table style="width:100%;border-collapse:collapse;background:#fff;'
        f'border:1px solid {LINE};border-radius:10px;overflow:hidden">'
        f'<thead>{head}</thead><tbody>{body}</tbody></table></div>'
    )


def results_note_html() -> str:
    return (
        f'<div style="font-family:{FONT};font-size:14px;color:{INK};'
        f'line-height:1.6;padding:2px">'
        f'<div><b>AUBC</b> là vùng dưới đường mAP theo % dữ liệu. Càng cao thì '
        f'gán nhãn càng hiệu quả.</div>'
        f'<div style="margin-top:6px">Bốn chiến lược chênh nhau dưới 1 mAP, gần như '
        f'ngang nhau. CoreSet thấp hơn cả Random.</div>'
        f'<div style="margin-top:6px;font-size:12.5px;color:{MUTED}">'
        f'Số liệu lấy từ <code>results/*/reports/al/</code>. Đây là benchmark đã lưu.'
        f'</div>'
        f'</div>'
    )
