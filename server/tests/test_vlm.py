import json
from unittest.mock import patch, MagicMock
from server.vlm import decide
from server.models import VLMAction


def _mock(response_text: str):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"response": response_text}
    r.raise_for_status = lambda: None
    return r


def test_decide_parses_arrived(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG marker
    payload = json.dumps({"action": "ARRIVED", "guidance": "milk on right",
                          "question": None, "vlm_summary": "dairy"})
    with patch("server.vlm.requests.post", return_value=_mock(payload)):
        resp = decide(image_path=str(img), goal="milk", goal_objects=["milk"],
                      topomap_summary="", detections_summary="milk x1",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.ARRIVED
    assert resp.guidance == "milk on right"
    assert resp.question is None


def test_decide_falls_back_on_unparseable(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    with patch("server.vlm.requests.post", return_value=_mock("hmm let me think...")):
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.MOVE
    assert "trouble" in resp.guidance.lower() or "another" in resp.guidance.lower()


def test_decide_falls_back_on_exception(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    with patch("server.vlm.requests.post", side_effect=Exception("boom")):
        resp = decide(image_path=str(img), goal="g", goal_objects=[],
                      topomap_summary="", detections_summary="",
                      prior_question=None, prior_answer=None)
    assert resp.action == VLMAction.MOVE


def test_decide_includes_prior_answer_block(tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    payload = json.dumps({"action": "MOVE", "guidance": "ok", "question": None, "vlm_summary": "s"})
    captured = {}

    def fake_post(url, **kw):
        captured["body"] = kw["json"]
        return _mock(payload)

    with patch("server.vlm.requests.post", side_effect=fake_post):
        decide(image_path=str(img), goal="g", goal_objects=[],
               topomap_summary="", detections_summary="",
               prior_question="Are you in dairy?", prior_answer="yes")
    assert "Are you in dairy?" in captured["body"]["prompt"]
    assert "yes" in captured["body"]["prompt"]
