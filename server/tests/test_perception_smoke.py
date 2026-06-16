"""Smoke test — only runs when SMOKE=1 env var is set, since it downloads the model."""
import os
import pytest
from pathlib import Path
from server.perception import Perception


@pytest.mark.skipif(os.environ.get("SMOKE") != "1", reason="downloads GroundingDINO from HF hub")
def test_perception_loads_and_detects():
    p = Perception()
    p.load()

    sample = Path("/home/user/UniGoal/assets/demo_real.gif")
    if not sample.exists():
        pytest.skip("no sample image")
    detections = p.detect(str(sample), ["person", "wall"])
    assert isinstance(detections, list)
