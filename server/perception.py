"""GroundingDINO (via Hugging Face transformers) perception. Loaded once at server startup."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

from server.config import (
    GROUNDINGDINO_MODEL_ID, SAM_WEIGHTS,
    GROUNDINGDINO_BOX_THRESHOLD, GROUNDINGDINO_TEXT_THRESHOLD, SAM_TOP_K_BOXES,
)

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    box: List[float]  # [x1,y1,x2,y2] absolute pixels
    score: float


class Perception:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = None
        self._gd_model = None
        self._sam_predictor = None

    def load(self) -> None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        log.info("loading GroundingDINO (transformers) %s", GROUNDINGDINO_MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(GROUNDINGDINO_MODEL_ID)
        self._gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDINGDINO_MODEL_ID
        ).to(self.device)
        self._gd_model.eval()
        # SAM is intentionally not loaded: detect() only needs GroundingDINO boxes,
        # and SAM ViT-H + llama3.2-vision together exceed the 8 GB VRAM budget.
        self._sam_predictor = None
        log.info("perception loaded on %s (GroundingDINO only)", self.device)

    def detect(self, image_path: str, prompt_classes: List[str]) -> List[Detection]:
        if not prompt_classes:
            return []
        text_prompt = ". ".join(prompt_classes) + "."
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._gd_model(**inputs)
        parsed = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=GROUNDINGDINO_BOX_THRESHOLD,
            text_threshold=GROUNDINGDINO_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1]],
        )[0]

        results: List[Detection] = [
            Detection(label=label, box=[float(c) for c in box.tolist()], score=float(score))
            for box, score, label in zip(parsed["boxes"], parsed["scores"], parsed["labels"])
        ]
        results.sort(key=lambda d: -d.score)
        return results[:SAM_TOP_K_BOXES]
