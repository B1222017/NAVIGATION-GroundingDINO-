"""Centralized prompt templates for goal decomposition and per-turn VLM calls."""

GOAL_DECOMPOSE_PROMPT = """\
You are helping someone navigate an indoor environment. Their goal is: "{goal}".
List 4-8 visual things they should look for to know they're close to the goal,
as a single comma-separated list. Be specific. Reply ONLY with the list, no preamble.

Example for "find the refrigerator":
refrigerator, fridge, kitchen area, coffee machine, water dispenser, food storage
"""


PER_TURN_PROMPT = """\
You are guiding a person through an indoor environment using their phone camera.

Goal: "{goal}"
Goal-related items to look for: {goal_objects}

What's happened so far:
{topomap_summary}

The person just uploaded the attached photo. In it, automatic detection found:
{detections_summary}

Text visible in the photo (OCR):
{ocr_summary}

{prior_answer_block}

Decide the next step. Reply with EXACTLY one JSON object on one line, nothing else:

{{"action": "ARRIVED" | "MOVE" | "ASK", "guidance": "<one or two sentences>", "question": "<short question, only if ASK, else null>", "vlm_summary": "<one phrase summarizing the location>"}}

Rules:
- ARRIVED only if the goal item is clearly visible in the photo (point at it in `guidance`).
- ASK if you cannot decide between two plausible directions and a yes/no answer would resolve it.
- MOVE otherwise. Tell the user a concrete direction (e.g., "turn left and walk down the corridor, then take another photo").
- Do NOT invent details not in the photo or detections.
- USE the OCR text to identify specific places, sections, aisle names, signs, and labels. The text tells you WHERE you are (e.g., "Dairy", "Aisle 5", "Exit", "Checkout").
"""


PRIOR_ANSWER_BLOCK = """\
The person just answered your earlier question "{previous_question}" with:
"{user_answer}"
"""
