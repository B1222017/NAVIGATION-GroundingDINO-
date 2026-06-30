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

# ── Navigation goal (set dynamically via --goal, never hardcode here) ─────

GOAL_TEXT    = ""   # "尋找{GOAL_TARGET}"
GOAL_TARGET  = ""   # the raw goal string, e.g. "電腦" or "吳世琳辦公室"
GOAL_PLACE   = "室內"
GOAL_OCR_NAME  = ""
GOAL_OCR_PARTS: list[str] = []
TARGET_KEYWORDS: set[str] = set()
SIMILAR_KEYWORDS: set[str] = set()   # intentionally empty — orange category removed
DETECT_CLASSES: list[str] = []

# Environment/landmark objects that are always detected (no goal dependency)
_BASE_CLASSES = [
    "door", "office door", "nameplate", "name plate", "door sign",
    "sign", "plaque", "room sign", "office sign",
    "bulletin board", "notice board", "whiteboard", "poster", "notice",
    "fire extinguisher", "trash can", "trash", "garbage can",
    "water dispenser", "water cooler", "water fountain",
    "refrigerator",
    "chair", "table", "desk", "counter", "reception desk", "reception",
    "cabinet", "bookshelf", "shelf", "locker", "printer",
    "plant", "sofa", "window", "elevator", "staircase",
]

# Known goal → detection classes + OCR confirmation keywords
_GOAL_CLASS_MAP: dict[str, dict] = {
    "電腦": {
        "classes":   ["電腦", "桌上型電腦", "筆記型電腦", "螢幕", "鍵盤", "滑鼠",
                      "computer", "laptop", "desktop computer", "monitor",
                      "keyboard", "mouse", "screen"],
        "ocr_parts": ["電腦", "桌機", "筆電", "本機", "computer", "PC", "desktop", "laptop"],
    },
    "冰箱": {
        "classes":   ["冰箱", "refrigerator", "fridge", "冷藏"],
        "ocr_parts": ["冰箱", "refrigerator", "fridge"],
    },
    "飲水機": {
        "classes":   ["飲水機", "water dispenser", "water cooler", "water fountain"],
        "ocr_parts": ["飲水機", "water dispenser"],
    },
    "印表機": {
        "classes":   ["印表機", "printer", "laser printer"],
        "ocr_parts": ["印表機", "printer"],
    },
    "辦公室": {
        "classes":   ["辦公室", "教師辦公室", "office", "nameplate", "name plate",
                      "door sign", "plaque", "office door"],
        "ocr_parts": [],  # augmented with person name at runtime
    },
}


def setup_goal(goal_text: str) -> None:
    """Derive all goal-dependent globals from the goal string."""
    global GOAL_TEXT, GOAL_TARGET, GOAL_OCR_NAME, GOAL_OCR_PARTS
    global TARGET_KEYWORDS, DETECT_CLASSES

    GOAL_TARGET   = goal_text
    GOAL_TEXT     = f"尋找{goal_text}"
    GOAL_OCR_NAME = goal_text

    # Look up known goal entries (one or more keys may match)
    extra_classes: list[str] = []
    extra_ocr: list[str] = []
    for key, info in _GOAL_CLASS_MAP.items():
        if key in goal_text:
            extra_classes.extend(info["classes"])
            extra_ocr.extend(info["ocr_parts"])

    # Always treat the raw goal text itself as a class and OCR token
    if goal_text not in extra_classes:
        extra_classes.insert(0, goal_text)
    if goal_text not in extra_ocr:
        extra_ocr.insert(0, goal_text)

    GOAL_OCR_PARTS  = list(dict.fromkeys(extra_ocr))     # dedup, preserve order
    TARGET_KEYWORDS = set(extra_classes)
    DETECT_CLASSES  = [c for c in extra_classes if c not in _BASE_CLASSES] + _BASE_CLASSES

ARRIVE_THRESHOLD  = 0.40   # GroundingDINO confidence for nameplate/door-sign detection
OCR_ARRIVE_CONF   = 0.40   # OCR confidence to trust name match as ARRIVED
VERIFY_THRESHOLD  = 0.30   # confidence to trigger informational VLM verification

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
        "goal": "",
        "observations": [],
        "best_goal_score": 0.0,
        "best_goal_step": None,
        "arrived": False,
        "false_positives": [],
        "corrections": [],
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
    Crop the detection box and ask VLM whether it matches the goal.
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
        f"問題：圖中有沒有「{GOAL_TARGET}」？\n"
        f"請用繁體中文回答，格式：\n"
        f"是否為目標：是/否\n"
        f"實際內容：[描述圖中內容]\n"
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
                             state: dict,
                             ocr_texts: list[str] | None = None) -> str:
    """
    Send current photo + text query to VLM and return navigation guidance.
    ocr_texts: list of OCR-recognized strings from current photo.
    """
    img_b64 = _encode_image(photo_path)

    def _det_label(d):
        ctx = d.get("context", "")
        return f"{d['label']}{'(' + ctx + ')' if ctx else ''}({d['score']:.2f})"

    det_str  = ", ".join(_det_label(d) for d in detections) or "無偵測結果"
    prev_str = ", ".join(f"{d['label']}({d['score']:.2f})" for d in prev_detections) or "無"
    ocr_str  = "、".join(ocr_texts) if ocr_texts else "（無可讀文字）"

    false_pos_note = ""
    if state.get("corrections"):
        false_pos_note = f"\n注意：{state['corrections'][-1]}"

    if user_text:
        prompt = (
            f"你是一個室內導航助手，正在幫助用戶尋找目標：{GOAL_TEXT}。\n\n"
            f"當前照片已附上。\n"
            f"偵測到的物件：{det_str}\n"
            f"上一步物件：{prev_str}\n"
            f"OCR 文字：{ocr_str}\n"
            f"{false_pos_note}\n\n"
            f"用戶問：「{user_text}」\n\n"
            f"請用繁體中文直接回答用戶的問題，根據照片內容和偵測結果給出具體答案。"
            f"不要忽略用戶的問題，不要只做場景介紹。回答控制在3句內。\n"
        )
    else:
        prompt = (
            f"你是一個室內導航助手，正在幫助用戶尋找目標：{GOAL_TEXT}。\n\n"
            f"當前步驟 {step} 的照片已附上。\n"
            f"當前偵測到的物件（含位置描述）：{det_str}\n"
            f"上一步偵測到的物件：{prev_str}\n"
            f"本張照片 OCR 讀出的文字：{ocr_str}\n"
            f"{false_pos_note}\n\n"
            f"請根據照片內容、偵測物件和文字辨識結果，用繁體中文回答：\n"
            f"1. 你看到了什麼（特別注意任何與「{GOAL_TARGET}」相關的物品或標示）\n"
            f"2. 是否看到「{GOAL_TARGET}」或任何相關線索\n"
            f"3. 具體的移動建議（往前/往左/往右/折返）\n"
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

def _norm_iou(a: list, b: list, wa: int, ha: int, wb: int, hb: int) -> float:
    """IoU between two boxes each given in absolute pixels of their respective images,
    normalized to [0,1] coordinates before comparison."""
    a_n = [a[0]/wa, a[1]/ha, a[2]/wa, a[3]/ha]
    b_n = [b[0]/wb, b[1]/hb, b[2]/wb, b[3]/hb]
    ix1, iy1 = max(a_n[0], b_n[0]), max(a_n[1], b_n[1])
    ix2, iy2 = min(a_n[2], b_n[2]), min(a_n[3], b_n[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a_n[2]-a_n[0])*(a_n[3]-a_n[1]) + (b_n[2]-b_n[0])*(b_n[3]-b_n[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _label_matches(a: str, b: str) -> bool:
    """True if labels refer to the same object class (handles compound labels like 'chair sofa')."""
    if a == b:
        return True
    # Allow match if one label is a subset of the other (e.g. 'sofa' ⊂ 'chair sofa')
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    return bool(a_words & b_words)   # any word in common


def tag_same_entity(detections: list[dict], img_w: int, img_h: int,
                    prev_dets: list[dict], prev_w: int, prev_h: int,
                    pos_thresh: float = 0.25) -> None:
    """
    Mark each detection as same_entity=True if it matches a detection in the
    PREVIOUS observation by label AND either:
      - Normalised bounding-box center is within pos_thresh (0.25), OR
      - Normalised IoU > 0.05 (boxes overlap in the image)
    Handles merged boxes (shifted centres) and compound labels ('chair sofa').
    """
    used_prev: set[int] = set()   # one-to-one: each prev detection can only match once
    for det in detections:
        nc = _norm_center(det["box"], img_w, img_h)
        best_idx, best_pd, best_dist = None, None, float("inf")
        for pi, pd in enumerate(prev_dets):
            if pi in used_prev:
                continue
            if not _label_matches(det["label"], pd["label"]):
                continue
            pnc = _norm_center(pd["box"], prev_w, prev_h)
            dx = abs(nc[0] - pnc[0])
            dy = abs(nc[1] - pnc[1])
            center_ok = dx < pos_thresh and dy < pos_thresh
            iou_ok = _norm_iou(det["box"], pd["box"], img_w, img_h, prev_w, prev_h) > 0.05
            if center_ok or iou_ok:
                dist = (dx**2 + dy**2) ** 0.5
                if dist < best_dist:
                    best_dist, best_idx, best_pd = dist, pi, pd
        if best_pd is not None:
            used_prev.add(best_idx)
            det["same_entity"] = True
            det["same_entity_prev_uid"] = best_pd.get("id", best_pd["label"])

def is_goal(label: str) -> bool:
    return any(kw in label.lower() for kw in TARGET_KEYWORDS)

def is_similar_to_goal(label: str) -> bool:
    return any(kw in label.lower() for kw in SIMILAR_KEYWORDS)

# ── NMS (deduplicate overlapping boxes) ───────────────────────────────────

def _iou(a: list, b: list) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0

def apply_nms(detections: list[dict], iou_thr: float = 0.50) -> list[dict]:
    """Remove lower-confidence boxes that overlap significantly with a higher-confidence one."""
    keep, suppressed = [], set()
    for i, d in enumerate(detections):   # already sorted by score desc
        if i in suppressed:
            continue
        keep.append(d)
        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            if _iou(d["box"], detections[j]["box"]) > iou_thr:
                suppressed.add(j)
    return keep


def _boxes_nearby(a: list, b: list, gap_ratio: float = 0.6) -> bool:
    """True if boxes overlap or the gap between them is small relative to their size."""
    h_gap = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    v_gap = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    min_side = min(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1])
    thr = gap_ratio * min_side
    return h_gap <= thr and v_gap <= thr


def merge_adjacent_same_label(detections: list[dict]) -> list[dict]:
    """
    Merge detections of the same label whose boxes are adjacent or overlapping.
    Keeps the highest-confidence score; bounding box becomes the union of merged boxes.
    """
    from collections import defaultdict
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_label[d["label"]].append(i)

    used = set()
    result = []

    for label, idxs in by_label.items():
        # Build clusters of nearby boxes (greedy single-linkage)
        clusters: list[list[int]] = []
        for idx in idxs:
            merged_into = None
            for cluster in clusters:
                if any(_boxes_nearby(detections[idx]["box"], detections[c]["box"]) for c in cluster):
                    cluster.append(idx)
                    merged_into = cluster
                    break
            if merged_into is None:
                clusters.append([idx])

        for cluster in clusters:
            if len(cluster) == 1:
                result.append(detections[cluster[0]])
            else:
                boxes = [detections[i]["box"] for i in cluster]
                merged_box = [
                    min(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), max(b[3] for b in boxes),
                ]
                best = max(cluster, key=lambda i: detections[i]["score"])
                merged_det = dict(detections[best])
                merged_det["box"] = [round(x, 1) for x in merged_box]
                result.append(merged_det)

    result.sort(key=lambda d: -d["score"])
    return result


def merge_adjacent_similar_label(detections: list[dict]) -> list[dict]:
    """Merge boxes with similar labels (per _label_matches) that are adjacent or overlapping.
    Handles cases like 'counter reception desk' + 'reception desk' detecting the same object."""
    n = len(detections)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if (_label_matches(detections[i]["label"], detections[j]["label"]) and
                    _boxes_nearby(detections[i]["box"], detections[j]["box"])):
                union(i, j)

    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    result = []
    for idxs in groups.values():
        if len(idxs) == 1:
            result.append(detections[idxs[0]])
        else:
            boxes = [detections[i]["box"] for i in idxs]
            merged_box = [
                min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes),
            ]
            best = max(idxs, key=lambda i: detections[i]["score"])
            merged_det = dict(detections[best])
            merged_det["box"] = [round(x, 1) for x in merged_box]
            result.append(merged_det)

    result.sort(key=lambda d: -d["score"])
    return result


def _nameplate_id(nameplate_text: str) -> str:
    """
    Extract a meaningful id from OCR nameplate text.
    Looks for sequences of 2+ consecutive Chinese characters (name + title).
    e.g. '陳仁暉 Ph.D. 助理教授' → '陳仁暉助理教授'
    """
    import re
    segments = re.findall(r'[一-鿿]{2,}', nameplate_text)
    if segments:
        # Join first 1-2 segments (likely name + title), cap at 8 chars
        return "".join(segments[:2])[:8]
    # Fallback: strip ASCII noise
    cleaned = re.sub(r'[A-Za-z0-9\s\.%\"\'_,\-]+', '', nameplate_text).strip()
    return cleaned[:8]


def assign_detection_ids(detections: list[dict]) -> None:
    """
    Add a unique 'id' field to each detection in-place.
    - Doors/signs with OCR nameplate text → use that text as id (e.g. '陳仁暉教授')
    - Single occurrences of a label keep the label as id
    - Duplicates without nameplate text get _1, _2, ... suffix
    """
    from collections import Counter
    label_count = Counter(d["label"] for d in detections)
    label_idx: dict[str, int] = {}
    used_ids: set[str] = set()

    for d in detections:
        lbl = d["label"]
        nameplate = d.get("nameplate_text", "")

        # Doors/nameplates with readable OCR text → use OCR as id
        if nameplate and any(kw in lbl.lower() for kw in _DOOR_LABELS | {"sign", "plaque", "nameplate"}):
            base_id = _nameplate_id(nameplate) or lbl   # fallback to label if no Chinese found
            if base_id and base_id not in used_ids:
                d["id"] = base_id
                used_ids.add(base_id)
                continue
            # Fallback: nameplate text is already used → append label suffix
            suffix = 1
            while f"{base_id}_{suffix}" in used_ids:
                suffix += 1
            d["id"] = f"{base_id}_{suffix}"
            used_ids.add(d["id"])
            continue

        # No nameplate: single label → keep as-is; duplicates → _N suffix
        if label_count[lbl] == 1:
            d["id"] = lbl
        else:
            label_idx[lbl] = label_idx.get(lbl, 0) + 1
            d["id"] = f"{lbl}_{label_idx[lbl]}"
        used_ids.add(d["id"])

# ── OCR ──────────────────────────────────────────────────────────────────

def _get_ocr_reader():
    if not hasattr(_get_ocr_reader, "_r"):
        import easyocr
        print("  [OCR] 初始化 EasyOCR（首次較慢）...")
        _get_ocr_reader._r = easyocr.Reader(["ch_tra", "en"], verbose=False)
    return _get_ocr_reader._r

def run_ocr_with_bbox(photo_path: str) -> list[tuple[list[float], str, float]]:
    """Run EasyOCR. Returns [(xyxy_box, text, confidence), ...] sorted by conf desc."""
    try:
        reader = _get_ocr_reader()
        raw = reader.readtext(str(photo_path))
        out = []
        for pts, text, conf in raw:
            text = text.strip()
            if not text or conf < 0.25:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            out.append(([float(min(xs)), float(min(ys)),
                         float(max(xs)), float(max(ys))], text, float(conf)))
        return sorted(out, key=lambda x: -x[2])
    except Exception as e:
        print(f"  [OCR] 失敗: {e}")
        return []

def run_ocr(photo_path: str) -> list[tuple[str, float]]:
    """Convenience wrapper — strips bboxes."""
    return [(t, c) for _, t, c in run_ocr_with_bbox(photo_path)]

def ocr_contains_name(ocr_results: list[tuple[str, float]]) -> tuple[bool, str, float]:
    """Check if OCR results contain the professor's name (or partial match).
    Returns (found, matched_text, confidence)."""
    best_text, best_conf = "", 0.0
    for text, conf in ocr_results:
        for part in GOAL_OCR_PARTS:
            if part in text and conf > best_conf:
                best_text, best_conf = text, conf
    if best_text:
        return True, best_text, best_conf
    return False, "", 0.0

# ── Spatial context inference ─────────────────────────────────────────────

_SURFACE_KEYWORDS = {"table", "desk", "cabinet", "counter", "shelf",
                     "sofa", "chair", "glass table", "windowsill"}

def infer_spatial_context(detections: list[dict]) -> None:
    """Add 'context' field to each detection describing what it sits on.
    E.g. plant → 'plant (在 cabinet 上)'.
    Modifies detections in-place."""
    for det in detections:
        det.setdefault("context", "")
        lbl_lower = det["label"].lower()
        # Surfaces don't need a context
        if any(s in lbl_lower for s in _SURFACE_KEYWORDS):
            continue
        x1a, y1a, x2a, y2a = det["box"]
        cx_a = (x1a + x2a) / 2
        best_surface, best_overlap = None, 0.0
        for other in detections:
            if other is det:
                continue
            if not any(s in other["label"].lower() for s in _SURFACE_KEYWORDS):
                continue
            x1b, y1b, x2b, y2b = other["box"]
            # A is "on" B if A's horizontal center is within B's horizontal range
            # and A's bottom is inside or just above B
            if not (x1b < cx_a < x2b):
                continue
            if not (y1b <= y2a <= y2b + (y2b - y1b) * 0.35):
                continue
            overlap = min(x2a, x2b) - max(x1a, x1b)
            if overlap > best_overlap:
                best_overlap = overlap
                best_surface = other["label"]
        if best_surface:
            det["context"] = f"在{best_surface}上"

# ── OCR → door spatial association ───────────────────────────────────────

_DOOR_LABELS = {"door", "office door"}

def associate_ocr_to_doors(detections: list[dict],
                            ocr_with_bbox: list[tuple[list, str, float]]) -> None:
    """
    For each door detection, collect OCR text regions that are horizontally
    adjacent (nameplate is mounted to the side of the door frame) AND at a
    similar vertical level.  Adds 'nameplate_text' / 'nameplate_conf' to each
    door detection so callers can display "door: 吳世琳 副教授" instead of "door".
    Modifies detections in-place.

    Key exclusions to avoid picking up posters/bulletin boards ON the door:
      - OCR box whose centre is mostly INSIDE the door bbox → skip (it's on the door)
      - OCR box wider than 50% of door width → too large to be a nameplate
      - OCR box taller than 30% of door height → same reason
    """
    for det in detections:
        det.setdefault("nameplate_text", "")
        det.setdefault("nameplate_conf", 0.0)
        if not any(kw in det["label"].lower() for kw in _DOOR_LABELS):
            continue
        dx1, dy1, dx2, dy2 = det["box"]
        door_h = dy2 - dy1
        door_w = dx2 - dx1

        texts, best_conf = [], 0.0
        for bbox, text, conf in ocr_with_bbox:
            ox1, oy1, ox2, oy2 = bbox
            ocr_w = ox2 - ox1
            ocr_h = oy2 - oy1
            ocr_cy = (oy1 + oy2) / 2

            # ── Reject text that is ON the door (poster / bulletin board) ──
            # If more than 50% of the OCR box overlaps with the door bbox, it's
            # printed on the door itself, not a nameplate beside the frame.
            overlap_x = max(0.0, min(ox2, dx2) - max(ox1, dx1))
            overlap_y = max(0.0, min(oy2, dy2) - max(oy1, dy1))
            overlap_ratio = (overlap_x * overlap_y) / max(ocr_w * ocr_h, 1)
            if overlap_ratio > 0.50:
                continue

            # ── Reject oversized text boxes (posters / large signs on door) ──
            if ocr_w > door_w * 0.50 or ocr_h > door_h * 0.30:
                continue

            # Nameplate is in the upper 70% of the door vertically
            v_ok = dy1 <= ocr_cy <= dy1 + door_h * 0.75
            # Horizontal gap between the two boxes (0 if they overlap horizontally)
            h_gap = max(0.0, max(ox1 - dx2, dx1 - ox2))
            # Within 1.5× door-width to the side — nameplates sit right beside frame
            if h_gap < door_w * 1.5 and v_ok and conf >= 0.35:
                texts.append(text)
                best_conf = max(best_conf, conf)

        if texts:
            det["nameplate_text"] = " ".join(texts)
            det["nameplate_conf"] = best_conf

# ── Annotated scene image ────────────────────────────────────────────────

def generate_scene_image(photo_path: str, detections: list[dict],
                          step: int, vlm_note: str = "",
                          ocr_with_bbox: list | None = None) -> Path:
    from PIL import Image, ImageDraw
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    def _load_cjk_font(size: int):
        from PIL import ImageFont
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default()

    # Scale proportionally to image width, using 1200px as the reference
    # (reference image has ~14px label font at 1200px wide)
    scale   = w / 1200
    fs      = max(12, int(14 * scale))   # main label font (~14px at 1200px wide)
    fs_sm   = max(10, int(11 * scale))   # OCR / banner font (~11px at 1200px wide)
    lh      = int(fs * 1.45)
    lh_sm   = int(fs_sm * 1.45)
    box_lw  = max(2, int(2 * scale))     # box line width (~2px at 1200px wide)
    char_w  = int(fs * 0.62)             # approx char width for bg rectangle

    _fnt    = _load_cjk_font(fs)
    _fnt_sm = _load_cjk_font(fs_sm)

    for d in detections:
        x1, y1, x2, y2 = [max(0, c) for c in d["box"]]
        x2, y2 = min(w, x2), min(h, y2)
        if is_goal(d["label"]):
            color = (220, 50, 50)
        elif is_similar_to_goal(d["label"]):
            color = (255, 140, 0)
        else:
            color = (50, 200, 80)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_lw)

        # Header label (unique id + confidence)
        nameplate = d.get("nameplate_text", "")
        hdr = f"{d.get('id', d['label'])} {d['score']:.2f}"
        tw = len(hdr) * char_w
        label_y0 = max(0, y1 - lh)
        draw.rectangle([x1, label_y0, x1 + tw, y1], fill=color)
        draw.text((x1 + int(4 * scale), label_y0 + int(3 * scale)),
                  hdr, fill=(255, 255, 255), font=_fnt)

        # Nameplate banner inside door box
        if nameplate and any(kw in d["label"].lower() for kw in _DOOR_LABELS):
            name_short = nameplate[:20]
            draw.rectangle([x1 + 2, y1 + 2, x2 - 2, y1 + lh_sm + 4], fill=(0, 0, 0))
            draw.text((x1 + int(6 * scale), y1 + int(4 * scale)),
                      name_short, fill=(255, 255, 100), font=_fnt_sm)

    # Draw OCR bounding boxes (cyan) with text + confidence
    if ocr_with_bbox:
        OCR_COLOR = (0, 200, 220)
        ocr_lw = max(2, int(4 * scale))
        for bbox, text, conf in ocr_with_bbox:
            ox1, oy1, ox2, oy2 = [int(c) for c in bbox]
            ox1, oy1 = max(0, ox1), max(0, oy1)
            ox2, oy2 = min(w, ox2), min(h, oy2)
            draw.rectangle([ox1, oy1, ox2, oy2], outline=OCR_COLOR, width=ocr_lw)
            label_txt = f'"{text[:14]}" {conf:.0%}'
            tw = len(label_txt) * int(fs_sm * 0.6)
            label_y0 = max(0, oy1 - lh_sm)
            draw.rectangle([ox1, label_y0, ox1 + tw, oy1], fill=OCR_COLOR)
            draw.text((ox1 + int(3 * scale), label_y0 + int(2 * scale)),
                      label_txt, fill=(0, 0, 0), font=_fnt_sm)

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

def _wrap_text(text: str, line_w: int) -> list[str]:
    """Wrap text to line_w chars, breaking at spaces when possible."""
    text = text.replace("\n", " ").strip()
    lines = []
    while len(text) > line_w:
        pos = text.rfind(" ", 0, line_w)
        if pos < line_w // 3:
            pos = line_w
        lines.append(text[:pos])
        text = text[pos:].lstrip()
    if text:
        lines.append(text)
    return lines or [""]


def _obs_box_h(obs: dict) -> float:
    """Compute obs node box height needed to display full VLM text."""
    interactions = obs.get("interactions") or [
        {"user_text": obs.get("user_text", ""), "vlm_guidance": obs.get("vlm_guidance", "")}
    ]
    n_int = len(interactions)
    HEADER_H = 0.62   # 觀察點 + photo rows
    LINE_H   = 0.20   # height per VLM text line
    PAD      = 0.15   # bottom padding

    if n_int == 1:
        inter  = interactions[0]
        has_ut = bool(inter.get("user_text", ""))
        vlm    = inter.get("vlm_guidance", "").replace("\n", " ")
        n_lines = len(_wrap_text(vlm, 22)) if vlm else 0
        return max(1.54, HEADER_H + (0.22 if has_ut else 0) + n_lines * LINE_H + PAD)
    else:
        # Parallel: compute per-sub-node height, use maximum
        sub_line_w = max(10, 22 // n_int)
        max_lines  = 0
        has_ut_any = False
        for inter in interactions:
            vlm    = inter.get("vlm_guidance", "").replace("\n", " ")
            n_lines = len(_wrap_text(vlm, sub_line_w)) if vlm else 0
            max_lines  = max(max_lines, n_lines)
            has_ut_any = has_ut_any or bool(inter.get("user_text", ""))
        return max(1.54, HEADER_H + (0.22 if has_ut_any else 0) + max_lines * LINE_H + PAD)


def _row_h_for(n_dets: int, is_current: bool = False, obs_box_h: float = 1.54) -> float:
    """Dynamic row height: enough to fit obs box AND detection nodes.

    The obs box top is fixed at y+0.77 and expands *downward* to y+0.77-box_h,
    so the row must satisfy:  rh >= 2*(box_h - 0.77) + gap  i.e. 2*box_h - 1.54 + gap.
    """
    if is_current:
        NODE_H, GAP = 0.58, 0.24
    else:
        NODE_H, GAP = 0.44, 0.20
    min_for_obs = max(2.20, 2 * obs_box_h - 1.54 + 0.50)
    return max(min_for_obs, n_dets * (NODE_H + GAP) + 0.70)


def render_scene_graph(observations: list[dict], step: int,
                        corrections: list[str]) -> Path:
    fp = _cjk()
    n_obs = len(observations)

    # Dynamic per-observation row heights so nodes never overlap
    row_heights = [
        _row_h_for(len(obs["detections"]), obs["step"] == step, _obs_box_h(obs))
        for obs in observations
    ]
    total_h = sum(row_heights)
    fig_h = max(6, total_h + 2.5)

    fig, ax = plt.subplots(figsize=(15, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    corr_note = f"  ⚠ {corrections[-1]}" if corrections else ""
    ax.set_title(
        f"場景圖 Scene Graph — 累積至步驟 {step}{corr_note}\n"
        f"(棕=正式找到  紅=目標候選  紫=同種非目標  灰=誤判  藍=同一實體  綠=地標)",
        fontsize=13, fontweight="bold", color="#222",
        fontproperties=fp, pad=8)

    obs_x = 1.8

    # Build cumulative obs_y positions from top down.
    # y_cursor = TOP edge of each row so rows never overlap regardless of height.
    # obs_y stores the ROW CENTER = top - rh/2.
    obs_y: dict[int, float] = {}
    y_cursor = fig_h - 1.3          # top edge of first row (below title)
    for i, obs in enumerate(observations):
        rh_i = row_heights[i]
        obs_y[obs["step"]] = y_cursor - rh_i / 2   # centre
        y_cursor -= rh_i                             # next row top = this row bottom

    node_data: list[dict] = []  # reserved for future use

    for i, obs in enumerate(observations):
        s = obs["step"]
        y = obs_y[s]
        rh = row_heights[i]
        is_current = (s == step)
        is_false_pos = obs.get("false_positive", False)

        fc = "#ffeeba" if is_false_pos else ("#d0e4f7" if is_current else "#e8f0f8")
        ec = "#ffc107" if is_false_pos else ("#5b9bd5" if is_current else "#a0b8cc")
        lw = 2.0 if is_current else (2.0 if is_false_pos else 1.2)

        # Backward-compatible: derive interactions list
        interactions = obs.get("interactions") or [
            {"user_text": obs.get("user_text", ""), "vlm_guidance": obs.get("vlm_guidance", "")}
        ]
        n_int    = len(interactions)
        ocr_hit  = any(p in t for t in obs.get("ocr_texts", []) for p in GOAL_OCR_PARTS)
        box_h    = _obs_box_h(obs)   # dynamic height to fit full VLM text
        box_top  = y + 0.77          # top is always fixed
        box_bot  = box_top - box_h

        # ── Horizontal parallel layout: n_int sub-nodes side by side ────
        # Right edge must stay at 3.30 to avoid overlapping history detection nodes (ox=5.0, left≈3.3)
        TOTAL_W      = 3.00
        X_LEFT       = 0.30
        GAP          = 0.10
        sub_w        = (TOTAL_W - GAP * (n_int - 1)) / n_int
        sub_xs       = [X_LEFT + k * (sub_w + GAP) for k in range(n_int)]
        sub_cxs      = [xl + sub_w / 2 for xl in sub_xs]
        box_center_y = box_top - box_h / 2
        group_right  = X_LEFT + TOTAL_W
        LINE_H       = 0.20    # vertical step per wrapped VLM line
        line_w       = max(10, int(sub_w / 0.118))   # chars per line ≈ sub_w / char_width

        label_txt = f"觀察點 {s}" + (" ⚠誤判" if is_false_pos else "")

        for k, inter in enumerate(interactions):
            sub_fc = fc if k == 0 else "#fff3e0"
            sub_ec = ec if k == 0 else "#e65100"
            sub_cx = sub_cxs[k]

            sub_box = FancyBboxPatch(
                (sub_xs[k], box_bot), sub_w, box_h,
                boxstyle="round,pad=0.06",
                facecolor=sub_fc, edgecolor=sub_ec, linewidth=lw, zorder=3)
            ax.add_patch(sub_box)

            # ── Header (top of box) ──
            suffix = "" if k == 0 else f" ＋{k}"
            ax.text(sub_cx, box_top - 0.20,
                    label_txt if k == 0 else f"觀察點 {s}{suffix}",
                    ha="center", va="center",
                    fontsize=9 if n_int > 1 else 12, fontproperties=fp,
                    color="#1a4a7a" if is_current else "#4a6a8a",
                    fontweight="bold" if is_current else "normal", zorder=4)
            if k == 0:
                ax.text(sub_cx, box_top - 0.38, obs["photo"],
                        ha="center", va="center", fontsize=7.5, color="#777", zorder=4)
            if k == 0 and ocr_hit:
                ax.text(sub_cx, box_top - 0.55, f"✅ OCR：{GOAL_OCR_NAME}",
                        ha="center", va="center", fontsize=7.5,
                        color="#155724", fontproperties=fp, zorder=4)

            # ── Content: user text then full VLM text ──
            y_cursor = box_top - 0.60 - (0.18 if (k == 0 and ocr_hit) else 0)

            ut = inter.get("user_text", "")
            if ut:
                ax.text(sub_cx, y_cursor, f'💬 「{ut}」',
                        ha="center", va="center", fontsize=7.5,
                        color="#a05000", fontproperties=fp, zorder=4)
                y_cursor -= 0.22

            vlm = inter.get("vlm_guidance", "")
            if vlm:
                wrapped = _wrap_text(vlm, line_w)
                ax.text(sub_cx, y_cursor, "🤖",
                        ha="center", va="center", fontsize=7, color="#444", zorder=4)
                y_cursor -= LINE_H * 0.6
                for line in wrapped:
                    ax.text(sub_cx, y_cursor, line,
                            ha="center", va="center", fontsize=7,
                            color="#333", fontproperties=fp, zorder=4)
                    y_cursor -= LINE_H

        # Arrow to next observation node (from group centre, dynamic box bottom)
        if i < n_obs - 1:
            ny = obs_y[observations[i + 1]["step"]]
            group_cx = (sub_cxs[0] + sub_cxs[-1]) / 2
            ax.annotate("", xy=(group_cx, ny + 0.77),
                        xytext=(group_cx, box_bot),
                        arrowprops=dict(arrowstyle="->", color="#7a9fc0", lw=1.3))

        # Object nodes
        # History:      x≈5.0
        # Current new:  x≈8.0  (new discoveries, not seen before)
        # Current same: x≈12.5 (same-entity, continuing from prev step)
        dets = obs["detections"]
        if not dets:
            continue

        n       = len(dets)
        spacing = rh / (n + 1)
        fs      = 11.0 if is_current else 8.0
        # node_h: tall enough for label + conf lines; never exceed spacing
        MIN_H  = 0.52 if is_current else 0.38
        node_h = max(MIN_H, min(MIN_H * 1.2, spacing - 0.14))
        # char_w_est: rough data-unit width per character at this fontsize
        char_w_est = 0.140 if is_current else 0.098

        for j, det in enumerate(dets):
            lbl         = det["label"]
            uid         = det.get("id", lbl)   # unique id
            ctx         = det.get("context", "")
            score       = det["score"]
            dy          = y + rh / 2 - spacing * (j + 1)
            same_entity = det.get("same_entity", False)

            # Per-node x: current same-entity → right (12.5), current new → middle (8.0)
            if is_current:
                ox = 12.5 if same_entity else 8.0
            else:
                ox = 5.0

            nameplate = det.get("nameplate_text", "")
            if nameplate and any(kw in lbl.lower() for kw in _DOOR_LABELS):
                display_lbl = f"{uid}:「{nameplate[:10]}」"
            elif ctx:
                ctx_s = ctx if len(ctx) <= 9 else ctx[:8] + "…"
                display_lbl = f"{uid} ({ctx_s})"
            else:
                display_lbl = uid
            # Hard-cap: ensures text always fits inside the node box
            _mc = 26 if is_current else 17
            if len(display_lbl) > _mc:
                display_lbl = display_lbl[:_mc - 1] + "…"

            # Dynamic node width: fit the label text, capped so it stays in-column
            # Current same-entity (x=12.5): wider; current new (x=8.0): narrower
            if is_current and not same_entity:
                node_w = min(3.6, max(2.2, len(display_lbl) * char_w_est + 0.55))
            else:
                node_w = min(
                    6.0 if is_current else 3.4,
                    max(4.2 if is_current else 2.3,
                        len(display_lbl) * char_w_est + 0.55)
                )

            lbl_for_color = lbl
            if (nameplate and any(kw in lbl.lower() for kw in _DOOR_LABELS)
                    and any(p in nameplate for p in GOAL_OCR_PARTS)):
                lbl_for_color = "nameplate"

            # ── Node colour by confirmed state ──────────────────────────
            # Brown  : 正式找到，導航結束
            # Gray   : 誤判（使用者確認與目標完全無關）
            # Purple : 同種非目標（使用者確認是同類但非目標）
            # Red    : 目標候選（尚待確認）
            # Blue   : 🔗 同一實體（跨步追蹤）
            # Green  : 一般地標
            if obs.get("arrived_here"):
                fc2, ec2, tc = "#d7ccc8", "#6d4c41", "#3e2723"   # brown
                tag = "✅"
            elif obs.get("false_positive"):
                fc2, ec2, tc = "#eeeeee", "#757575", "#424242"   # gray — 誤判
                tag = "❌"
            elif obs.get("wrong_instance"):
                fc2, ec2, tc = "#e8d5f5", "#7b1fa2", "#4a0072"  # purple — 同種非目標
                tag = "↩"
            elif is_goal(lbl_for_color):
                fc2, ec2, tc = "#f8d7da", "#dc3545", "#721c24"   # red — 目標候選
                tag = "🎯"
            elif same_entity:
                fc2, ec2, tc = "#cce5ff", "#004085", "#003060"   # blue — 同一實體
                tag = "🔗"
            else:
                fc2, ec2, tc = "#d4edda", "#28a745", "#155724"   # green — 地標
                tag = ""

            nb = FancyBboxPatch((ox - node_w / 2, dy - node_h / 2), node_w, node_h,
                                boxstyle="round,pad=0.05",
                                facecolor=fc2, edgecolor=ec2,
                                linewidth=1.4 if is_current else 0.8,
                                alpha=1.0 if is_current else 0.6, zorder=2)
            ax.add_patch(nb)

            label_y = dy + node_h * 0.18
            conf_y  = dy - node_h * 0.22
            ax.text(ox, label_y, f"{tag} {display_lbl}",
                    ha="center", va="center", fontsize=fs,
                    color=tc, fontproperties=fp, zorder=3)
            ax.text(ox, conf_y, f"conf: {score:.3f}",
                    ha="center", va="center", fontsize=fs - 1.5,
                    color="#999", zorder=3)

            # Arrow: obs node group → object node
            edge_color = "#5b9bd5" if is_current else "#ccc"
            edge_lw    = 0.9 if is_current else 0.4
            ax.annotate("", xy=(ox - node_w / 2, dy),
                        xytext=(group_right, box_center_y),
                        arrowprops=dict(arrowstyle="->", color=edge_color, lw=edge_lw))

            node_data.append({"label": lbl, "uid": uid, "step": s, "x": ox, "y": dy,
                               "node_w": node_w, "surface": ctx,
                               "same_entity": same_entity})

    # ── Cross-step same-entity edges ─────────────────────────────────────
    # Use right-bowing arcs (arc3, negative rad = bows right for downward arrows).
    # Each unique label gets an increasingly wider arc so curves never overlap.
    # No elbow segments → no crossing lines.
    _label_idx: dict[str, int] = {}
    _COLOURS = ["#004085", "#7b1fa2", "#1a7a40", "#b34700", "#005f73", "#8b0000"]

    for i, obs in enumerate(observations[1:], 1):
        prev_step_n = observations[i - 1]["step"]
        curr_step_n = obs["step"]
        for det in obs["detections"]:
            if not det.get("same_entity"):
                continue
            det_uid = det.get("id", det["label"])
            prev_uid = det.get("same_entity_prev_uid", det["label"])
            prev_n = next((n for n in node_data
                           if n["step"] == prev_step_n
                           and n["uid"] == prev_uid), None)
            curr_n = next((n for n in node_data
                           if n["step"] == curr_step_n
                           and n["uid"] == det_uid), None)
            if not (prev_n and curr_n):
                continue

            lbl_key = det["label"]
            if lbl_key not in _label_idx:
                _label_idx[lbl_key] = len(_label_idx)
            idx    = _label_idx[lbl_key]
            colour = _COLOURS[idx % len(_COLOURS)]

            sx = prev_n["x"] + prev_n["node_w"] / 2
            sy = prev_n["y"]
            ex = curr_n["x"] - curr_n["node_w"] / 2
            ey = curr_n["y"]

            same_col = abs(prev_n["x"] - curr_n["x"]) < 1.0
            if same_col:
                # Right-bowing arc; negative rad bows rightward for downward arrows.
                # Wider arc per label index so they fan out and don't overlap.
                rad = -(0.30 + idx * 0.18)
            else:
                # Cross-column: leftward gentle curve so the arc stays in view
                rad = -(0.08 + idx * 0.04)

            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.8,
                                        linestyle="dashed", alpha=0.90,
                                        connectionstyle=f"arc3,rad={rad:.2f}"),
                        zorder=5)

    # ── Spatial edges "在…上": arcs on RIGHT side of current step nodes ──
    curr_obs_obj = next((o for o in observations if o["step"] == step), None)
    if curr_obs_obj:
        for det in curr_obs_obj["detections"]:
            ctx = det.get("context", "")
            if not ctx:
                continue
            surface_lbl = ctx.replace("在", "").replace("上", "").strip()
            curr_n = next((n for n in node_data
                           if n["step"] == step and n["label"] == det["label"]), None)
            surf_n = next((n for n in node_data
                           if n["step"] == step
                           and surface_lbl in n["label"].lower()), None)
            if curr_n and surf_n:
                ax.annotate("",
                    xy=(surf_n["x"] + surf_n["node_w"] / 2, surf_n["y"]),
                    xytext=(curr_n["x"] + curr_n["node_w"] / 2, curr_n["y"]),
                    arrowprops=dict(arrowstyle="->", color="#1a7a40", lw=1.2,
                                    linestyle="dotted", alpha=0.90,
                                    connectionstyle="arc3,rad=0.4"),
                    zorder=5)
                mid_x = max(curr_n["x"], surf_n["x"]) + curr_n["node_w"] / 2 + 0.3
                mid_y = (curr_n["y"] + surf_n["y"]) / 2
                ax.text(mid_x, mid_y, "在…上", fontsize=7,
                        color="#1a7a40", ha="left", va="center",
                        fontproperties=fp, zorder=6)

    from matplotlib.lines import Line2D
    legend_items = [
        mpatches.Patch(facecolor="#d7ccc8", edgecolor="#6d4c41", label="✅ 正式找到（導航結束）"),
        mpatches.Patch(facecolor="#f8d7da", edgecolor="#dc3545", label="🎯 目標候選（待確認）"),
        mpatches.Patch(facecolor="#e8d5f5", edgecolor="#7b1fa2", label="↩ 同種非目標（用戶確認）"),
        mpatches.Patch(facecolor="#eeeeee", edgecolor="#757575", label="❌ 誤判（與目標完全無關）"),
        mpatches.Patch(facecolor="#cce5ff", edgecolor="#004085", label="🔗 同一實體（跨步追蹤）"),
        mpatches.Patch(facecolor="#d4edda", edgecolor="#28a745", label="地標物件"),
        Line2D([0], [0], color="#004085", lw=1.2, linestyle="dashed",
               label="- - 跨步同一實體追蹤"),
        Line2D([0], [0], color="#1a7a40", lw=1.0, linestyle="dotted",
               label="··· 空間關係（在…上）"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=10,
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

def print_table(photo: str, detections: list[dict], prev_labels: set,
                ocr_results: list[tuple[str, float]] | None = None) -> None:
    print(f"\n{'─'*72}")
    print(f"  📷  {photo}")
    print(f"{'─'*72}")
    if not detections:
        print("  ⚠  No objects detected")
    else:
        print(f"  {'#':<3} {'Label + 位置描述':<38} {'Conf':>6}  Entity  Type")
        print(f"  {'─'*3} {'─'*38} {'─'*6}  {'─'*6}  {'─'*22}")
        for i, d in enumerate(detections, 1):
            uid = d.get("id", d["label"])   # unique id (e.g. door_1, door_2)
            lbl = d["label"]
            ctx = d.get("context", "")
            display_lbl = f"{uid} ({ctx})" if ctx else uid
            if d.get("same_entity"):
                sh = "🔗同一"
            elif lbl in prev_labels:
                sh = "↔同類"
            else:
                sh = "     "
            if is_goal(lbl):
                tag = "🎯 目標候選"
            else:
                tag = "   地標"
            print(f"  {i:<3} {display_lbl:<38} {d['score']:>6.3f}  {sh}  {tag}")
            if d.get("nameplate_text") and any(kw in lbl.lower() for kw in _DOOR_LABELS):
                name_hit = any(p in d["nameplate_text"] for p in GOAL_OCR_PARTS)
                marker = "  ⭐ 目標！" if name_hit else ""
                print(f"       └ 門旁文字：「{d['nameplate_text']}」"
                      f"(conf={d.get('nameplate_conf', 0):.2f}){marker}")
    # OCR results section
    if ocr_results is not None:
        print(f"  {'─'*70}")
        print(f"  🔤 OCR 文字辨識：")
        if ocr_results:
            for text, conf in ocr_results[:8]:
                name_hit = any(p in text for p in GOAL_OCR_PARTS)
                marker = "  ✅ 名字符合！" if name_hit else ""
                print(f"     「{text}」  conf={conf:.2f}{marker}")
        else:
            print("     （未讀到文字）")
    print(f"{'─'*72}")

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

# ── Terminal input helper ─────────────────────────────────────────────────

def _ask_user(prompt_text: str) -> str:
    """Read input from terminal, supporting interactive, batch, and Colab modes."""
    # Colab replaces sys.stdin with a widget that supports input() directly
    try:
        import google.colab  # noqa: F401
        return input(prompt_text).strip().lower()
    except ImportError:
        pass
    try:
        if sys.stdin.isatty():
            return input(prompt_text).strip().lower()
        # stdin is piped/redirected — open controlling terminal directly
        with open("/dev/tty") as tty:
            sys.stdout.write(prompt_text)
            sys.stdout.flush()
            return tty.readline().strip().lower()
    except (EOFError, OSError, KeyboardInterrupt):
        return ""

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", help="Path to the photo for this step")
    parser.add_argument("--text", default="", help="Text description / question from user")
    parser.add_argument("--goal", default="", help="Navigation goal, e.g. '電腦' or '冰箱'")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--correct", action="store_true",
                        help="Mark last ARRIVED as wrong. Add --same if same type but wrong instance.")
    parser.add_argument("--same", action="store_true",
                        help="Used with --correct: same type of object but wrong specific one")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Reset ──────────────────────────────────────────────────────────
    if args.reset:
        if Path(SESSION_JSON).exists():
            Path(SESSION_JSON).unlink()
        print("✅ Session reset.")
        return

    state = load_session()

    # ── Setup goal (from --goal arg, or restore from saved session) ────
    goal_text = args.goal or state.get("goal", "")
    if not goal_text:
        print("❌ 請用 --goal 指定導航目標，例如：--goal 電腦")
        return
    setup_goal(goal_text)
    state["goal"] = goal_text  # persist so --correct doesn't need --goal

    # ── Correct: reject last ARRIVED ───────────────────────────────────
    if args.correct:
        if not state.get("arrived"):
            print("ℹ  沒有待修正的 ARRIVED 狀態。")
            return
        state["arrived"] = False
        last_arrived_step = state.get("best_goal_step")

        if args.same:
            # Purple: 同種非目標 — right category, wrong instance
            for obs in state["observations"]:
                if obs["step"] == last_arrived_step:
                    obs["wrong_instance"] = True
                    obs["arrived_here"] = False
            correction_msg = (f"步驟 {last_arrived_step}：是同種目標（{GOAL_TARGET}），"
                              f"但不是要找的那個，繼續搜索")
            state["corrections"].append(correction_msg)
            state["best_goal_score"] = 0.0
            state["best_goal_step"] = None
            state["best_ocr_found"] = False
            save_session(state)
            print(f"✅ 了解，是同種（{GOAL_TARGET}）但非目標。場景圖標為紫色。")
            print(f"   請提供下一張照片。")
        else:
            # Gray: 誤判 — completely unrelated to the goal
            if last_arrived_step is not None:
                state["false_positives"].append(last_arrived_step)
                for obs in state["observations"]:
                    if obs["step"] == last_arrived_step:
                        obs["false_positive"] = True
                        obs["arrived_here"] = False
            correction_msg = f"步驟 {last_arrived_step} 的目標宣告是誤判（與目標完全無關），繼續搜索 {GOAL_TARGET}"
            state["corrections"].append(correction_msg)
            state["best_goal_score"] = 0.0
            state["best_goal_step"] = None
            state["best_ocr_found"] = False
            save_session(state)
            print(f"✅ 已標記步驟 {last_arrived_step} 為誤判（完全無關）。場景圖標為灰色。")
            print(f"   繼續搜索：請提供下一張照片。")

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

    photo_name = photo_path.name
    user_text = args.text.strip()

    # ── Same-photo: merge interaction into existing node ───────────────
    last_obs = state["observations"][-1] if state["observations"] else None
    if last_obs is not None and last_obs["photo"] == photo_name:
        step = last_obs["step"]
        # Init interactions list on older obs records that predate this field
        if "interactions" not in last_obs:
            last_obs["interactions"] = [
                {"user_text": last_obs.get("user_text", ""),
                 "vlm_guidance": last_obs.get("vlm_guidance", "")}
            ]
        vlm_guidance = ""
        if user_text:
            print(f"\n🔁 同一位置（{photo_name}），合併至觀察點 {step}")
            print(f"  用戶說：「{user_text}」")
            print(f"\n[VLM] 根據新提問重新詢問模型...")
            prev_dets = []
            vlm_guidance = ask_navigation_guidance(
                str(photo_path), last_obs["detections"], prev_dets, user_text,
                step, state, ocr_texts=last_obs.get("ocr_texts", []))
            print(f"\n  💬 VLM 回應：\n  {vlm_guidance}\n")
        else:
            print(f"\n🔁 同一位置（{photo_name}），合併至觀察點 {step}（無新提問）")
        last_obs["interactions"].append({"user_text": user_text, "vlm_guidance": vlm_guidance})
        save_session(state)
        sg_path = render_scene_graph(state["observations"], step, state.get("corrections", []))
        print(f"[場景圖] 更新 → {sg_path}")
        return

    state["step"] += 1
    step = state["step"]

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
    before_nms = len(detections)
    detections = apply_nms(detections, iou_thr=0.50)
    detections = merge_adjacent_same_label(detections)
    detections = merge_adjacent_similar_label(detections)
    after_merge = len(detections)
    if before_nms != after_merge:
        print(f"    NMS+Merge: {before_nms} → {after_merge} 個框")
    if prev_obs:
        pw, ph = prev_obs.get("img_size", [4032, 3024])
        tag_same_entity(detections, img_w, img_h,
                        prev_obs["detections"], pw, ph)
    infer_spatial_context(detections)

    # 2. OCR — run first, then associate to doors, then assign ids
    print(f"\n[2] EasyOCR 文字辨識...")
    ocr_with_bbox = run_ocr_with_bbox(str(photo_path))
    ocr_results = [(t, c) for _, t, c in ocr_with_bbox]
    ocr_texts = [t for t, _ in ocr_results]
    print(f"    讀到 {len(ocr_results)} 段文字")
    associate_ocr_to_doors(detections, ocr_with_bbox)
    assign_detection_ids(detections)   # after OCR so nameplate_text is available

    ocr_name_found, ocr_matched_text, ocr_name_conf = ocr_contains_name(ocr_results)
    if ocr_name_found:
        print(f"    ✅ OCR 發現名字符合：「{ocr_matched_text}」（conf={ocr_name_conf:.2f}）")

    print_table(photo_name, detections, prev_labels, ocr_results)

    # 3. VLM: navigation guidance (if user provided text)
    vlm_guidance = ""
    if user_text:
        print(f"\n[3] VLM 導航指引（根據照片 + 用戶文字 + OCR）...")
        vlm_guidance = ask_navigation_guidance(
            str(photo_path), detections, prev_dets, user_text, step, state,
            ocr_texts=ocr_texts)
        print(f"\n  💬 VLM 回應：\n  {vlm_guidance}\n")
    else:
        print(f"\n[3] 未輸入文字，跳過 VLM 導航指引。")

    # 4. Annotated scene image
    print(f"\n[4] 生成標注圖像...")
    vlm_note = vlm_guidance[:80] if vlm_guidance else ""
    scene_path = generate_scene_image(str(photo_path), detections, step, vlm_note,
                                       ocr_with_bbox=ocr_with_bbox)
    print(f"    → {scene_path}")

    # 5. Goal graph
    print(f"\n[5] 渲染目標圖...")
    goal_path = render_goal_graph(step)
    print(f"    → {goal_path}")

    # Record observation
    obs_record = {
        "step": step,
        "photo": photo_name,
        "detections": detections,
        "ocr_texts": ocr_texts,
        "user_text": user_text,
        "vlm_guidance": vlm_guidance,
        "interactions": [{"user_text": user_text, "vlm_guidance": vlm_guidance}],
        "false_positive": False,
        "wrong_instance": False,
        "arrived_here": False,
        "img_size": [img_w, img_h],
    }
    state["observations"].append(obs_record)

    # 6. Scene graph
    print(f"\n[6] 渲染累積場景圖（{step} 個觀察點）...")
    sg_path = render_scene_graph(state["observations"], step, state.get("corrections", []))
    print(f"    → {sg_path}")

    save_session(state)

    # 7. ARRIVED decision
    # Primary: OCR found the professor's name at sufficient confidence
    # Secondary: GroundingDINO detected a nameplate/sign (informational VLM check)
    goal_dets = [d for d in detections if is_goal(d["label"])]
    vlm_verify_msg = ""

    # Check if any door has the target name in its associated nameplate text
    door_with_target = next(
        (d for d in detections
         if any(kw in d["label"].lower() for kw in _DOOR_LABELS)
         and any(p in d.get("nameplate_text", "") for p in GOAL_OCR_PARTS)
         and d.get("nameplate_conf", 0) >= OCR_ARRIVE_CONF),
        None
    )
    if door_with_target and not ocr_name_found:
        # Treat door-linked match same as global OCR match
        ocr_name_found = True
        ocr_matched_text = door_with_target["nameplate_text"]
        ocr_name_conf = door_with_target["nameplate_conf"]
        print(f"    ✅ 門牌綁定發現名字：「{ocr_matched_text}」→ 「{door_with_target['label']}」旁")

    print(f"\n{'='*65}")
    print(f"  STEP {step} 結果")

    # Summarise what was detected
    if ocr_name_found and ocr_name_conf >= OCR_ARRIVE_CONF:
        print(f"  ✅ OCR 主要訊號：找到「{ocr_matched_text}」(conf={ocr_name_conf:.2f})")
        if goal_dets:
            best = max(goal_dets, key=lambda d: d["score"])
            print(f"  🎯 GroundingDINO 補充：{best['label']} conf={best['score']:.3f}")
    elif goal_dets:
        best = max(goal_dets, key=lambda d: d["score"])
        print(f"  🎯 GroundingDINO 偵測到：{best['label']} conf={best['score']:.3f}")
        if ocr_name_found:
            print(f"  ✅ OCR 補充：「{ocr_matched_text}」(conf={ocr_name_conf:.2f})")
    else:
        print(f"  目標偵測：❌ 未偵測到「{GOAL_TARGET}」，OCR 亦無相關文字")

    shared = {d["label"] for d in detections} & prev_labels
    if shared:
        print(f"  與上步共享物件：{', '.join(sorted(shared))}")

    backtrack = check_backtrack(state, detections)
    print(f"{'='*65}")

    # Update best_goal_score for backtrack hints
    false_pos_steps = set(state.get("false_positives", []))
    if step not in false_pos_steps:
        if ocr_name_found and ocr_name_conf > state.get("best_goal_score", 0.0):
            state["best_goal_score"] = ocr_name_conf
            state["best_goal_step"] = step
            state["best_ocr_found"] = True
        elif goal_dets:
            best_s = max(d["score"] for d in goal_dets)
            if best_s > state.get("best_goal_score", 0.0) and not state.get("best_ocr_found"):
                state["best_goal_score"] = best_s
                state["best_goal_step"] = step

    # ── Interactive confirmation whenever ANY goal candidate is detected ──
    need_confirm = bool(goal_dets) or (ocr_name_found and ocr_name_conf >= OCR_ARRIVE_CONF)

    if need_confirm:
        best_det = max(goal_dets, key=lambda d: d["score"]) if goal_dets else None
        print(f"\n❓ 偵測到可能是「{GOAL_TARGET}」的物品：")
        if best_det:
            print(f"   • {best_det['label']}  conf={best_det['score']:.3f}")
        if ocr_name_found:
            print(f"   • OCR：「{ocr_matched_text}」 (conf={ocr_name_conf:.2f})")
        print(f"\n   [y] 是的，這就是我要找的！（導航完成）")
        print(f"   [s] 是同種物品，但不是要找的那個")
        print(f"   [n] 跟目標完全無關（誤判）")
        print(f"   [Enter] 跳過，繼續導航")
        answer = _ask_user("   你的選擇: ")

        if answer == "y":
            obs_record["arrived_here"] = True
            state["arrived"] = True
            save_session(state)
            print(f"\n🏁 ✅ 導航完成！已找到「{GOAL_TARGET}」")
        elif answer == "s":
            obs_record["wrong_instance"] = True
            state["corrections"].append(
                f"步驟 {step}：是同種目標（{GOAL_TARGET}），但不是要找的那個，繼續搜索")
            state["best_goal_score"] = 0.0
            state["best_goal_step"] = None
            save_session(state)
            print(f"↩ 了解，標記為紫色（同種非目標），繼續搜索。")
        elif answer == "n":
            obs_record["false_positive"] = True
            state["false_positives"].append(step)
            state["corrections"].append(
                f"步驟 {step} 的偵測是誤判（與目標完全無關），繼續搜索 {GOAL_TARGET}")
            state["best_goal_score"] = 0.0
            state["best_goal_step"] = None
            save_session(state)
            print(f"❌ 了解，標記為灰色（誤判），繼續搜索。")
        else:
            save_session(state)
            print(f"⏭ 跳過，繼續導航。")
    elif backtrack:
        print(f"\n{backtrack}")
        save_session(state)
    else:
        print(f"\n  ➡  未找到「{GOAL_TARGET}」，繼續前進。")
        save_session(state)

    for p in [scene_path, goal_path, sg_path]:
        subprocess.Popen(["open", str(p)])


if __name__ == "__main__":
    main()
