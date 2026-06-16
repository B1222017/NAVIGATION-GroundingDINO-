"""VLM client: send (image + prompt) to Ollama and parse structured response."""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import List, Optional

import requests

from server.config import OLLAMA_URL, OLLAMA_MODEL, VLM_TIMEOUT_S
from server.models import VLMAction, VLMResponse
from server.prompts import PER_TURN_PROMPT, PRIOR_ANSWER_BLOCK

log = logging.getLogger(__name__)

_FALLBACK = VLMResponse(
    action=VLMAction.MOVE,
    guidance="I had trouble understanding the scene. Try walking forward a few steps and uploading another photo.",
    question=None,
    vlm_summary="",
)


def _build_prompt(
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
) -> str:
    if prior_question and prior_answer:
        block = PRIOR_ANSWER_BLOCK.format(previous_question=prior_question, user_answer=prior_answer)
    else:
        block = ""
    return PER_TURN_PROMPT.format(
        goal=goal,
        goal_objects=", ".join(goal_objects) or "(none)",
        topomap_summary=topomap_summary or "(starting location)",
        detections_summary=detections_summary or "(no detections)",
        ocr_summary=ocr_summary or "(no text detected)",
        prior_answer_block=block,
    )


def _parse(text: str) -> Optional[VLMResponse]:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return VLMResponse(
            action=VLMAction(obj["action"]),
            guidance=str(obj.get("guidance", "")),
            question=obj.get("question"),
            vlm_summary=str(obj.get("vlm_summary", "")),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        log.warning("VLM JSON parse failed: %s", e)
        return None


def decide(
    image_path: str,
    goal: str,
    goal_objects: List[str],
    topomap_summary: str,
    detections_summary: str,
    prior_question: Optional[str],
    prior_answer: Optional[str],
    ocr_summary: Optional[str] = None,
) -> VLMResponse:
    prompt = _build_prompt(goal, goal_objects, topomap_summary, detections_summary, prior_question, prior_answer, ocr_summary=ocr_summary)
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "images": [img_b64], "stream": False},
            timeout=VLM_TIMEOUT_S,
        )
        r.raise_for_status()
        text = r.json().get("response", "")
    except Exception as e:
        log.warning("VLM call failed: %s", e)
        return _FALLBACK

    parsed = _parse(text)
    if parsed is None:
        log.warning("VLM unparseable response: %.200s", text)
        return _FALLBACK
    return parsed
