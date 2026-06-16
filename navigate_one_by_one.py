"""
navigate_one_by_one.py — single-photo navigation with VLM guidance.

Usage:
    python navigate_one_by_one.py --reset
    python navigate_one_by_one.py --photo IMG_1432.jpg
    python navigate_one_by_one.py --photo IMG_1434.jpg --text "我往前走了2步"
    python navigate_one_by_one.py --photo IMG_1436.jpg --text "我現在在走廊中間，應該往哪走？"
    python navigate_one_by_one.py --correct   # ARRIVED 是誤判，折返繼續
    python navigate_one_by_one.py --photo IMG_1438.jpg --text "我剛才看到的是飲水機，已折返"

Outputs per step (navigate_output/):
    stepNN_scene_IMGXXXX.jpg   — annotated photo
    stepNN_goal_graph.png      — goal graph
    stepNN_scene_graph.png     — cumulative scene graph (all obs chained)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Navigation goal ──────────────────────────────────────────────────────

GOAL_TEXT    = "尋找系辦內的冰箱"
GOAL_TARGET  = "冰箱"
GOAL_PLACE   = "系辦"

DETECT_CLASSES = [
    "refrigerator", "fridge", "freezer",
    "vending machine", "water dispenser", "water cooler", "microwave",
    "door", "sign", "fire extinguisher",
    "sofa", "chair", "table", "cabinet", "desk",
    "plant", "counter", "bulletin board",
]

TARGET_KEYWORDS  = {"refrigerator", "fridge", "freezer"}
# Keywords that look like goal but are NOT (flagged for VLM verification)
SIMILAR_KEYWORDS = {"vending machine", "water dispenser", "water cooler"}

ARRIVE_THRESHOLD  = 0.45   # confidence to trigger ARRIVED
VERIFY_THRESHOLD  = 0.35   # confidence to trigger VLM verification (informational) before ARRIVED

OLLAMA_URL        = "http://127.0.0.1:11434"
OLLAMA_MODEL      = "gemma4:latest"        # vision model (llama3.2-vision unsupported arch)
OLLAMA_TEXT_MODEL = "llama3.2:3b"          # text-only fallback
SESSION_JSON = "session_navigate.json"
OUTPUT_DIR   = Path("navigate_output")

# ── Session ──────────────────────────────────────────────────────────────

def load_session() -> dict:
    if Path(SESSION_JSON).exists():
        with open(SESSION_JSON) as f:
            return json.load(f)
    return {
        "step": 0,
        "observations": [],
        "best_goal_score": 0.0,
        "best_goal_step": None,
        "arrived": False,
        "false_positives": [],   # steps where ARRIVED was wrong
        "corrections": [],       # user correction messages
    }

def save_session(state: dict) -> None:
    with open(SESSION_JSON, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

# ── Ollama VLM ───────────────────────────────────────────────────────────

def _encode_image(path_or_pil) -> str:
    """Return base64-encoded JPEG string from file path or PIL image."""
    from PIL import Image
    if isinstance(path_or_pil, str) or isinstance(path_or_pil, Path):
        with open(str(path_or_pil), "rb") as f:
            return base64.b64encode(f.read()).decode()
    else:
        buf = io.BytesIO()
        path_or_pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

def ask_vlm(image_b64: str, prompt: str, timeout: int = 60) -> str:
    """Call Ollama vision model. Returns raw text or empty string on failure."""
    import requests
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt,
                  "images": [image_b64], "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"  [VLM] 呼叫失敗: {e}")
        return ""

def ask_vlm_text_only(prompt: str, timeout: int = 45) -> str:
    """Call Ollama text-only model."""
    import requests
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_TEXT_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"  [VLM text] 呼叫失敗: {e}")
        return ""

# ── VLM: verify goal detection ────────────────────────────────────────────

def verify_goal_with_vlm(photo_path: str, box: list[float],
                          goal_label: str) -> tuple[bool, str]:
    """
    Crop the detection box and ask VLM whether it's actually the goal.
    Returns (is_correct, vlm_explanation).
    """
    from PIL import Image
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = [max(0, int(c)) for c in box]
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return False, "裁切框太小，無法驗證"

    cropped = img.crop((x1, y1, x2, y2))
    img_b64 = _encode_image(cropped)

    prompt = (
        f"請仔細看這張裁切圖片。\n"
        f"問題：這個物品是「{goal_label}（冰箱/refrigerator）」嗎？\n"
        f"如果不是，它最可能是什麼？\n"
        f"請用繁體中文回答，格式：\n"
        f"是否為目標：是/否\n"
        f"實際物品：[物品名稱]\n"
        f"理由：[一句話說明]\n"
    )
    print(f"\n  [VLM 驗證] 裁切框 ({x1},{y1})→({x2},{y2})，詢問模型...")
    resp = ask_vlm(img_b64, prompt, timeout=60)
    if not resp:
        return True, "(VLM 無回應，採信偵測結果)"

    print(f"  [VLM 驗證回應]\n  {resp}\n")
    is_correct = "是" in resp.split("是否為目標：")[-1][:5] if "是否為目標" in resp else ("否" not in resp[:30])
    return is_correct, resp

# ── VLM: navigation guidance ──────────────────────────────────────────────

def ask_navigation_guidance(photo_path: str,
                             detections: list[dict],
                             prev_detections: list[dict],
                             user_text: str,
                             step: int,
                             state: dict) -> str:
    """
    Send current photo + text query to VLM and return navigation guidance.
    """
    img_b64 = _encode_image(photo_path)

    det_str = ", ".join(f"{d['label']}({d['score']:.2f})" for d in detections) or "無偵測結果"
    prev_str = ", ".join(f"{d['label']}({d['score']:.2f})" for d in prev_detections) or "無"

    false_pos_note = ""
    if state.get("false_positives"):
        fp_steps = state["false_positives"]
        false_pos_note = f"\n注意：步驟 {fp_steps} 的「抵達」宣告已被用戶確認為誤判（飲水機非冰箱），目前在繼續搜索。"

    prompt = (
        f"你是一個室內導航助手，正在幫助用戶尋找目標：{GOAL_TEXT}。\n\n"
        f"當前步驟 {step} 的照片已附上。\n"
        f"當前偵測到的物件：{det_str}\n"
        f"上一步偵測到的物件：{prev_str}\n"
        f"{false_pos_note}\n"
        f"用戶說：「{user_text}」\n\n"
        f"請根據照片內容和用戶的描述，用繁體中文回答：\n"
        f"1. 你看到了什麼（根據照片）\n"
        f"2. 目前最可能的冰箱位置判斷\n"
        f"3. 具體的移動建議（往前/往左/往右/折返/停止）\n"
        f"請直接給出建議，不要說「根據照片」等冗詞，回答控制在3句內。\n"
    )

    print(f"\n  [VLM 導航] 詢問模型...")
    resp = ask_vlm(img_b64, prompt, timeout=90)
    return resp if resp else "(VLM 無回應，請根據目視判斷方向)"

# ── Detection ────────────────────────────────────────────────────────────

def _get_perception():
    if not hasattr(_get_perception, "_p"):
        sys.path.insert(0, str(Path(__file__).parent))
        from server.perception import Perception
        p = Perception()
        p.load()
        _get_perception._p = p
    return _get_perception._p

def run_detection(photo_path: str) -> list[dict]:
    p = _get_perception()
    dets = p.detect(photo_path, DETECT_CLASSES)
    return [
        {"label": d.label, "score": round(d.score, 3),
         "box": [round(x, 1) for x in d.box],
         "same_entity": False}   # filled in by tag_same_entity() after detection
        for d in dets
    ]

def _norm_center(box: list, img_w: int, img_h: int) -> tuple[float, float]:
    """Return normalized (cx, cy) in [0,1] range for spatial comparison."""
    return ((box[0] + box[2]) / (2 * img_w),
            (box[1] + box[3]) / (2 * img_h))

def tag_same_entity(detections: list[dict], img_w: int, img_h: int,
                    prev_dets: list[dict], prev_w: int, prev_h: int,
                    pos_thresh: float = 0.15) -> None:
    """
    Mark each detection as same_entity=True if it matches a detection in the
    PREVIOUS observation by label AND normalized bounding-box center distance.
    'Same entity' means the same physical object, re-photographed from a nearby
    angle — NOT just the same object category.
    """
    for det in detections:
        nc = _norm_center(det["box"], img_w, img_h)
        for pd in prev_dets:
            if pd["label"] == det["label"]:
                pnc = _norm_center(pd["box"], prev_w, prev_h)
                if abs(nc[0] - pnc[0]) < pos_thresh and abs(nc[1] - pnc[1]) < pos_thresh:
                    det["same_entity"] = True
                    break

def is_goal(label: str) -> bool:
    return any(kw in label.lower() for kw in TARGET_KEYWORDS)

def is_similar_to_goal(label: str) -> bool:
    return any(kw in label.lower() for kw in SIMILAR_KEYWORDS)

# ── Annotated scene image ────────────────────────────────────────────────

def generate_scene_image(photo_path: str, detections: list[dict],
                          step: int, vlm_note: str = "") -> Path:
    from PIL import Image, ImageDraw
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    for d in detections:
        x1, y1, x2, y2 = [max(0, c) for c in d["box"]]
        x2, y2 = min(w, x2), min(h, y2)
        if is_goal(d["label"]):
            color = (220, 50, 50)
        elif is_similar_to_goal(d["label"]):
            color = (255, 140, 0)    # orange — similar but not goal
        else:
            color = (50, 200, 80)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        txt = f"{d['label']} {d['score']:.2f}"
        tw = len(txt) * 7
        draw.rectangle([x1, max(0, y1 - 22), x1 + tw, y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - 20)), txt, fill=(255, 255, 255))

    # VLM note banner at bottom
    if vlm_note:
        from PIL import ImageFont
        banner_h = 40
        canvas = Image.new("RGB", (w, h + banner_h), (30, 30, 30))
        canvas.paste(img, (0, 0))
        draw2 = ImageDraw.Draw(canvas)
        draw2.text((8, h + 6), f"VLM: {vlm_note[:100]}", fill=(255, 220, 50))
        img = canvas

    out = OUTPUT_DIR / f"step{step:02d}_scene_{Path(photo_path).stem}.jpg"
    img.save(str(out), "JPEG", quality=90)
    return out

# ── CJK font ─────────────────────────────────────────────────────────────

def _cjk():
    from matplotlib.font_manager import FontProperties, fontManager
    available = {f.name for f in fontManager.ttflist}
    for name in ["Heiti TC", "LiHei Pro", "Hiragino Sans GB",
                 "Hiragino Sans CNS", "Arial Unicode MS", "Hiragino Sans"]:
        if name in available:
            return FontProperties(family=name)
    return FontProperties()

# ── Goal Graph ────────────────────────────────────────────────────────────

def render_goal_graph(step: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 9); ax.set_ylim(0, 5)
    ax.axis("off")
    fp = _cjk()
    ax.set_title(f"目標圖 Goal Graph\n目標文字：{GOAL_TEXT}",
                 fontsize=13, fontweight="bold", color="#222",
                 fontproperties=fp, pad=10)

    def node(cx, cy, lines, type_lbl, fc, tc="#333", ec="#aaa"):
        box = FancyBboxPatch((cx - 1.1, cy - 0.48), 2.2, 0.96,
                             boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        for i, ln in enumerate(lines):
            ax.text(cx, cy + 0.12 - i * 0.26, ln,
                    ha="center", va="center", fontsize=10,
                    color=tc, fontproperties=fp, zorder=3)
        ax.text(cx, cy - 0.35, f"(type: {type_lbl})",
                ha="center", fontsize=7.5, color="#888", zorder=3)

    def arrow(x1, y1, x2, y2, lbl=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5))
        if lbl:
            ax.text((x1+x2)/2+0.1, (y1+y2)/2, lbl,
                    fontsize=8, color="#666", ha="left", va="center")

    lns = [GOAL_TEXT[:10], GOAL_TEXT[10:]] if len(GOAL_TEXT) > 10 else [GOAL_TEXT]
    node(4.5, 2.5, lns, "goal", "#d0e4f7", "#1a4a7a", "#5b9bd5")
    node(4.5, 4.2, [GOAL_PLACE], "place", "#d4edda", "#155724", "#28a745")
    node(7.5, 1.2, [GOAL_TARGET], "target", "#f8d7da", "#721c24", "#dc3545")
    arrow(4.5, 3.0, 4.5, 3.75, "located_in")
    arrow(5.3, 2.1, 6.4, 1.5, "target_of")

    out = OUTPUT_DIR / f"step{step:02d}_goal_graph.png"
    import contextlib, io as _io
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        plt.tight_layout()
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out

# ── Cumulative Scene Graph ────────────────────────────────────────────────

def render_scene_graph(observations: list[dict], step: int,
                        corrections: list[str]) -> Path:
    fp = _cjk()
    n_obs = len(observations)
    row_h = 2.2
    fig_h = max(6, n_obs * row_h + 2.0)

    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    corr_note = f"  ⚠ 用戶修正：{corrections[-1]}" if corrections else ""
    ax.set_title(
        f"場景圖 Scene Graph — 累積至步驟 {step}{corr_note}\n"
        f"(紅框=目標候選  橘框=相似但非目標  黃色=與前觀察點共享  虛線=跨觀察點同標籤)",
        fontsize=10, fontweight="bold", color="#222",
        fontproperties=fp, pad=8)

    obs_x = 1.8

    obs_y = {}
    for i, obs in enumerate(observations):
        obs_y[obs["step"]] = fig_h - 1.2 - i * row_h

    label_sets = {obs["step"]: {d["label"] for d in obs["detections"]}
                  for obs in observations}

    node_data: list[dict] = []  # unused after arc-line removal; kept for future use

    for i, obs in enumerate(observations):
        s = obs["step"]
        y = obs_y[s]
        is_current = (s == step)
        is_false_pos = obs.get("false_positive", False)

        fc = "#ffeeba" if is_false_pos else ("#d0e4f7" if is_current else "#e8f0f8")
        ec = "#ffc107" if is_false_pos else ("#5b9bd5" if is_current else "#a0b8cc")
        lw = 2.0 if is_current else (2.0 if is_false_pos else 1.2)

        box = FancyBboxPatch((obs_x - 1.3, y - 0.45), 2.6, 0.9,
                             boxstyle="round,pad=0.08",
                             facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(box)

        label_txt = f"觀察點 {s}"
        if is_false_pos:
            label_txt += " ⚠誤判"
        ax.text(obs_x, y + 0.1, label_txt, ha="center", va="center",
                fontsize=9, color="#1a4a7a" if is_current else "#4a6a8a",
                fontproperties=fp,
                fontweight="bold" if is_current else "normal", zorder=4)
        ax.text(obs_x, y - 0.22, obs["photo"],
                ha="center", va="center", fontsize=6.5, color="#777", zorder=4)
        if obs.get("user_text"):
            ax.text(obs_x, y - 0.38, f'💬 "{obs["user_text"][:28]}"',
                    ha="center", va="center", fontsize=6, color="#a05000", zorder=4)

        if i < n_obs - 1:
            ny = obs_y[observations[i+1]["step"]]
            ax.annotate("", xy=(obs_x, ny + 0.47),
                        xytext=(obs_x, y - 0.47),
                        arrowprops=dict(arrowstyle="->", color="#7a9fc0", lw=1.3))

        # Object nodes
        dets = obs["detections"]
        if not dets:
            continue

        ox = 9.5 if is_current else 6.2
        spacing = row_h / (len(dets) + 1)
        obs_img_size = obs.get("img_size", [4032, 3024])

        for j, det in enumerate(dets):
            lbl = det["label"]
            score = det["score"]
            dy = y + row_h/2 - spacing * (j + 1)
            same_entity = det.get("same_entity", False)

            if is_goal(lbl):
                if obs.get("false_positive"):
                    fc2, ec2, tc = "#ffe0b2", "#ff6d00", "#bf360c"
                    tag = "⚠️❌"
                else:
                    fc2, ec2, tc = "#f8d7da", "#dc3545", "#721c24"
                    tag = "🎯"
            elif is_similar_to_goal(lbl):
                fc2, ec2, tc = "#ffe0b2", "#ff9800", "#7f4400"
                tag = "⚠"
            elif same_entity:
                # Same physical object: same label + bbox center close to previous step
                fc2, ec2, tc = "#cce5ff", "#004085", "#003060"
                tag = "🔗"
            else:
                fc2, ec2, tc = "#d4edda", "#28a745", "#155724"
                tag = ""

            node_w = 4.6 if is_current else 3.2
            node_h = 0.38 if is_current else 0.28

            nb = FancyBboxPatch((ox - node_w/2, dy - node_h/2), node_w, node_h,
                                boxstyle="round,pad=0.06",
                                facecolor=fc2, edgecolor=ec2,
                                linewidth=1.5 if is_current else 0.8,
                                alpha=1.0 if is_current else 0.55, zorder=2)
            ax.add_patch(nb)
            fs = 8.5 if is_current else 6.5
            ax.text(ox, dy + 0.04, f"{tag} {lbl}",
                    ha="center", va="center", fontsize=fs, color=tc, zorder=3)
            ax.text(ox, dy - 0.14, f"conf: {score:.3f}",
                    ha="center", va="center", fontsize=fs - 1.5, color="#999", zorder=3)

            ax.annotate("", xy=(ox - node_w/2, dy),
                        xytext=(obs_x + 1.3, y),
                        arrowprops=dict(arrowstyle="->",
                                        color="#5b9bd5" if is_current else "#ccc",
                                        lw=1.0 if is_current else 0.5))
            if is_current:
                mx = (obs_x + 1.3 + ox - node_w/2) / 2
                ax.text(mx, (y + dy)/2 + 0.07, "mentioned_or_visible",
                        fontsize=5.5, color="#bbb", ha="center")

            node_data.append({
                "label": lbl, "obs_idx": i, "step": s,
                "box": det["box"], "img_size": obs_img_size,
                "x": ox, "y": dy,
            })

    legend_items = [
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#dc3545", label="🎯 目標候選"),
        mpatches.Patch(facecolor="#ffe0b2", edgecolor="#ff9800", label="⚠ 相似但非目標（如飲水機）"),
        mpatches.Patch(facecolor="#cce5ff", edgecolor="#004085",
                       label="🔗 同一物品（同標籤＋位置相符，同一實體）"),
        mpatches.Patch(facecolor="#d4edda", edgecolor="#28a745", label="新偵測物件"),
        mpatches.Patch(facecolor="#ffeeba", edgecolor="#ffc107", label="⚠ 誤判觀察點"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=7,
              facecolor="white", edgecolor="#ccc", prop=fp)

    out = OUTPUT_DIR / f"step{step:02d}_scene_graph.png"
    import contextlib, io as _io
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        plt.tight_layout()
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out

# ── Console table ─────────────────────────────────────────────────────────

def print_table(photo: str, detections: list[dict], prev_labels: set) -> None:
    print(f"\n{'─'*65}")
    print(f"  📷  {photo}")
    print(f"{'─'*65}")
    if not detections:
        print("  ⚠  No objects detected")
        return
    print(f"  {'#':<3} {'Label':<30} {'Conf':>6}  Entity  Type")
    print(f"  {'─'*3} {'─'*30} {'─'*6}  {'─'*6}  {'─'*22}")
    for i, d in enumerate(detections, 1):
        lbl = d["label"]
        if d.get("same_entity"):
            sh = "🔗同一"   # same physical object (label + position match)
        elif lbl in prev_labels:
            sh = "↔同類"    # same category only, different position
        else:
            sh = "     "
        if is_goal(lbl):
            tag = "🎯 TARGET CANDIDATE"
        elif is_similar_to_goal(lbl):
            tag = "⚠  similar (verify!)"
        else:
            tag = "   landmark"
        print(f"  {i:<3} {lbl:<30} {d['score']:>6.3f}  {sh}  {tag}")
    print(f"{'─'*65}")

# ── Backtrack check ───────────────────────────────────────────────────────

def check_backtrack(state: dict, current_dets: list[dict]) -> str | None:
    cur_best = max((d["score"] for d in current_dets if is_goal(d["label"])), default=0.0)
    hist_best = state.get("best_goal_score", 0.0)
    hist_step = state.get("best_goal_step")
    if hist_best >= 0.35 and cur_best < 0.20 and hist_step:
        obs = next((o for o in state["observations"] if o["step"] == hist_step), None)
        photo = obs["photo"] if obs else f"步驟{hist_step}"
        return (f"⚠️  折返提示：「{GOAL_TARGET}」曾在步驟 {hist_step} ({photo}) "
                f"出現（conf {hist_best:.3f}）但現在消失。\n"
                f"   建議折返至步驟 {hist_step} 重新確認。")
    return None

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", help="Path to the photo for this step")
    parser.add_argument("--text", default="", help="Text description / question from user")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--correct", action="store_true",
                        help="Mark last ARRIVED as wrong and continue searching")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Reset ──────────────────────────────────────────────────────────
    if args.reset:
        if Path(SESSION_JSON).exists():
            Path(SESSION_JSON).unlink()
        print("✅ Session reset.")
        return

    state = load_session()

    # ── Correct: reject last ARRIVED ───────────────────────────────────
    if args.correct:
        if not state.get("arrived"):
            print("ℹ  沒有待修正的 ARRIVED 狀態。")
            return
        state["arrived"] = False
        last_arrived_step = state.get("best_goal_step")
        if last_arrived_step is not None:
            state["false_positives"].append(last_arrived_step)
            for obs in state["observations"]:
                if obs["step"] == last_arrived_step:
                    obs["false_positive"] = True
        correction_msg = f"步驟 {last_arrived_step} 的目標宣告是誤判，繼續搜索 {GOAL_TARGET}"
        state["corrections"].append(correction_msg)
        state["best_goal_score"] = 0.0
        state["best_goal_step"] = None
        save_session(state)
        print(f"✅ 已標記步驟 {last_arrived_step} 為誤判（飲水機等非目標物）。")
        print(f"   繼續搜索：請提供下一張照片。")
        # Re-render scene graph to show false positive marking
        if state["observations"]:
            sg = render_scene_graph(state["observations"],
                                    state["step"], state["corrections"])
            print(f"   場景圖已更新：{sg}")
            subprocess.Popen(["open", str(sg)])
        return

    # ── Normal step: process photo ──────────────────────────────────────
    if not args.photo:
        parser.error("--photo <path> is required (or --reset / --correct)")

    photo_path = Path(os.path.expanduser(args.photo))
    if not photo_path.exists():
        print(f"❌ Photo not found: {photo_path}")
        sys.exit(1)

    if state.get("arrived") and not state.get("corrections"):
        print("🏁 已宣告抵達。如果是誤判，請執行：\n   python navigate_one_by_one.py --correct")
        return

    state["step"] += 1
    step = state["step"]
    photo_name = photo_path.name
    user_text = args.text.strip()

    print(f"\n{'='*65}")
    print(f"  STEP {step}  |  {photo_name}")
    if user_text:
        print(f"  用戶說：「{user_text}」")
    print(f"  Goal: {GOAL_TEXT}")
    print(f"{'='*65}")

    prev_obs = state["observations"][-1] if state["observations"] else None
    prev_labels = {d["label"] for d in prev_obs["detections"]} if prev_obs else set()
    prev_dets = prev_obs["detections"] if prev_obs else []

    # 1. Detection
    print(f"\n[1] GroundingDINO 偵測...")
    from PIL import Image as _PIL
    with _PIL.open(photo_path) as _im:
        img_w, img_h = _im.size
    detections = run_detection(str(photo_path))
    if prev_obs:
        pw, ph = prev_obs.get("img_size", [4032, 3024])
        tag_same_entity(detections, img_w, img_h,
                        prev_obs["detections"], pw, ph)
    print_table(photo_name, detections, prev_labels)

    # 2. VLM: navigation guidance (if user provided text)
    vlm_guidance = ""
    if user_text:
        print(f"\n[2] VLM 導航指引（根據照片 + 用戶文字）...")
        vlm_guidance = ask_navigation_guidance(
            str(photo_path), detections, prev_dets, user_text, step, state)
        print(f"\n  💬 VLM 回應：\n  {vlm_guidance}\n")
    else:
        print(f"\n[2] 未輸入文字，跳過 VLM 導航指引。")

    # 3. Annotated scene image
    print(f"\n[3] 生成標注圖像...")
    vlm_note = vlm_guidance[:80] if vlm_guidance else ""
    scene_path = generate_scene_image(str(photo_path), detections, step, vlm_note)
    print(f"    → {scene_path}")

    # 4. Goal graph
    print(f"\n[4] 渲染目標圖...")
    goal_path = render_goal_graph(step)
    print(f"    → {goal_path}")

    # Record observation
    obs_record = {
        "step": step,
        "photo": photo_name,
        "detections": detections,
        "user_text": user_text,
        "vlm_guidance": vlm_guidance,
        "false_positive": False,
        "img_size": [img_w, img_h],
    }
    state["observations"].append(obs_record)

    # best_goal_score is updated AFTER VLM verification (see below) so denied
    # detections don't trigger false backtrack hints.

    # 5. Scene graph
    print(f"\n[5] 渲染累積場景圖（{step} 個觀察點）...")
    sg_path = render_scene_graph(state["observations"], step, state.get("corrections", []))
    print(f"    → {sg_path}")

    save_session(state)

    # 6. Goal detection + VLM verification
    goal_dets = [d for d in detections if is_goal(d["label"])]
    vlm_verified = False
    vlm_verify_msg = ""

    print(f"\n{'='*65}")
    print(f"  STEP {step} 結果")

    if goal_dets:
        best = max(goal_dets, key=lambda d: d["score"])
        print(f"  🎯 目標偵測：{best['label']} @ conf {best['score']:.3f}")

        # VLM verification is informational only — shown to user but does NOT block ARRIVED.
        # If the detection is wrong, user runs --correct.
        if best["score"] >= VERIFY_THRESHOLD:
            print(f"\n[6] VLM 參考驗證（不影響判定）...")
            vlm_verified, verify_resp = verify_goal_with_vlm(
                str(photo_path), best["box"], GOAL_TARGET)
            vlm_verify_msg = verify_resp
            verdict = "✅ VLM 參考：看起來是冰箱類設備" if vlm_verified else f"⚠ VLM 參考：{verify_resp[:80]}"
            print(f"  {verdict}")

    else:
        print(f"  目標偵測：❌ 未偵測到 {GOAL_TARGET}")

    shared = {d["label"] for d in detections} & prev_labels
    if shared:
        print(f"  與上步共享物件：{', '.join(sorted(shared))}")

    backtrack = check_backtrack(state, detections)

    print(f"{'='*65}")

    # Update best_goal_score for any goal detection (VLM is informational only).
    false_pos_steps = set(state.get("false_positives", []))
    if step not in false_pos_steps:
        for d in detections:
            if is_goal(d["label"]) and d["score"] > state.get("best_goal_score", 0.0):
                state["best_goal_score"] = d["score"]
                state["best_goal_step"] = step

    # Decision: GroundingDINO score is the gate for ARRIVED.
    # VLM verification is informational only. User runs --correct to reject.
    if goal_dets and max(d["score"] for d in goal_dets) >= ARRIVE_THRESHOLD:
        best = max(goal_dets, key=lambda d: d["score"])
        print(f"\n🏁 早停 ARRIVED：{best['label']} conf={best['score']:.3f}")
        print(f"   如果認為是誤判，請執行：\n   python navigate_one_by_one.py --correct")
        state["arrived"] = True
        save_session(state)
    elif backtrack:
        print(f"\n{backtrack}")
    else:
        print(f"\n  ➡  目標未找到，繼續前進。")

    for p in [scene_path, goal_path, sg_path]:
        subprocess.Popen(["open", str(p)])


if __name__ == "__main__":
    main()
