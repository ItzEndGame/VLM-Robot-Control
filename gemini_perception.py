"""
Perception step: face-camera image + instruction -> which object to pick.

This is deliberately a SEPARATE module from the sim, so you can test it in
isolation (no PyBullet, no GUI needed) — just point it at any saved
scene.png and an instruction string.

Install:
    pip install google-genai pillow

Auth:
    Get an API key from https://aistudio.google.com/apikey and either:
      - set it as an environment variable:  setx GEMINI_API_KEY "..."
        (Windows, then open a NEW terminal) or export GEMINI_API_KEY=...
      - or pass --api-key on the command line below.

Standalone test:
    python gemini_perception.py --image scene.png --instruction "pick up the green ball"

As a library:
    from gemini_perception import query_target_object
    result = query_target_object("scene.png", "pick up the green ball",
                                  known_objects=["green_ball", "red_ball"])
    # result == {"target_object": "green_ball", "visible": True, "reasoning": "..."}

Design notes:
- Gemini is asked to pick from a fixed list of KNOWN object names, not to
  invent free-form labels or return pixel coordinates. The sim already
  knows each known object's real (x, y, z) from the physics engine, so
  the vision model's actual job here is just "which one is the person
  talking about", not "where exactly is it in pixel space". That keeps
  the JSON contract tiny and the parsing trivial — no bounding-box /
  depth back-projection math needed for this assignment's scope.
- response_mime_type="application/json" is used so Gemini is constrained
  to return valid JSON directly (no markdown code fences to strip).
- If the call fails for ANY reason (no API key, no network, bad JSON,
  model says the object isn't visible), this raises a PerceptionError —
  callers should catch that and fall back to a safe default rather than
  crashing the whole robot.
"""

import argparse
import json
import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current working directory (if present)
except ImportError:
    pass  # python-dotenv not installed — GEMINI_API_KEY must be set some other way


SEARCH_ACTIONS = ["turn_left", "turn_right", "move_forward", "move_backward", "target_reached"]


class PerceptionError(Exception):
    """Raised when Gemini perception fails or gives an unusable answer."""


def decide_next_action(image_path, instruction, known_objects, api_key=None, model="gemini-3.6-flash"):
    """
    One step of a closed-loop visual search: given the robot's CURRENT
    camera view + instruction, decide the single next discrete action —
    instead of the robot driving to one hardcoded point and hoping the
    target happens to be in frame there.

    Returns a dict: {"action": one of SEARCH_ACTIONS, "target_object": str|None,
                      "visible": bool, "reasoning": str}
    Raises PerceptionError on any failure.

    Caller is expected to execute the action, capture a new frame, and
    call this again — repeat until action == "target_reached" or a step
    budget runs out.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise PerceptionError("google-genai is not installed. Run: pip install google-genai") from e
    try:
        from PIL import Image
    except ImportError as e:
        raise PerceptionError("Pillow is not installed. Run: pip install pillow") from e

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PerceptionError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment "
            "variable or pass api_key=... explicitly."
        )
    if not os.path.exists(image_path):
        raise PerceptionError(f"Image not found: {os.path.abspath(image_path)}")
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise PerceptionError(f"Could not open image '{image_path}': {e}") from e

    prompt = f"""You are the navigation brain of a mobile robot searching for
an object. Its front camera just captured the attached image. Its
instruction is:

    "{instruction}"

Known pickable objects that might be somewhere in the scene: {known_objects}

Decide the SINGLE best next action for the robot, from exactly this set:
{SEARCH_ACTIONS}

Guidance:
- "turn_left" / "turn_right": use when the target isn't visible yet, to scan
  around for it.
- "move_forward": use when the target IS visible but still small/far away.
- "move_backward": use when the target is visible but too close, or cut off
  by the edges of the frame.
- "target_reached": use ONLY when the target named in the instruction is
  clearly visible, roughly centered, and close enough for a robot arm to
  reach it (not tiny or far away).

Respond with ONLY a JSON object of exactly this shape:
{{"action": "<one of the actions above>",
  "target_object": "<one of the known object names this instruction refers to, or null>",
  "visible": <true or false>,
  "reasoning": "<one short sentence>"}}"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception as e:
        raise PerceptionError(f"Gemini API call failed: {type(e).__name__}: {e}") from e

    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, AttributeError) as e:
        raise PerceptionError(
            f"Gemini did not return valid JSON. Raw response: {getattr(response, 'text', response)!r}"
        ) from e

    action = result.get("action")
    if action not in SEARCH_ACTIONS:
        raise PerceptionError(f"Gemini returned an invalid action: {action!r}")
    if action == "target_reached" and result.get("target_object") not in known_objects:
        raise PerceptionError(
            f"Gemini said target_reached but target_object={result.get('target_object')!r} "
            f"isn't in known_objects={known_objects}"
        )

    return result


def query_target_object(image_path, instruction, known_objects, api_key=None, model="gemini-3.6-flash"):
    """
    Send the captured image + instruction to Gemini and ask it to choose
    which of `known_objects` the instruction refers to.

    Returns a dict: {"target_object": str, "visible": bool, "reasoning": str}
    Raises PerceptionError on any failure (missing key, bad response, etc.)
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise PerceptionError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from e

    try:
        from PIL import Image
    except ImportError as e:
        raise PerceptionError("Pillow is not installed. Run: pip install pillow") from e

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PerceptionError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment "
            "variable or pass api_key=... explicitly."
        )

    if not os.path.exists(image_path):
        raise PerceptionError(f"Image not found: {os.path.abspath(image_path)}")

    try:
        image = Image.open(image_path)
    except Exception as e:
        raise PerceptionError(f"Could not open image '{image_path}': {e}") from e

    prompt = f"""You are the perception module of a mobile robot. The robot's
front camera just captured the attached image. The robot has been given
this instruction:

    "{instruction}"

Known pickable objects currently in the scene: {known_objects}

Look at the image and decide which single object from the known list the
instruction is asking the robot to pick up. If the instruction doesn't
clearly match any known object, or the target isn't visible in the image,
say so honestly rather than guessing.

Respond with ONLY a JSON object of exactly this shape:
{{"target_object": "<one of the known object names, or null>",
  "visible": <true or false>,
  "reasoning": "<one short sentence>"}}"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception as e:
        raise PerceptionError(f"Gemini API call failed: {type(e).__name__}: {e}") from e

    try:
        result = json.loads(response.text)
    except (json.JSONDecodeError, AttributeError) as e:
        raise PerceptionError(
            f"Gemini did not return valid JSON. Raw response: {getattr(response, 'text', response)!r}"
        ) from e

    target = result.get("target_object")
    if target is None:
        raise PerceptionError(
            f"Gemini couldn't match the instruction to a known object. "
            f"Reasoning: {result.get('reasoning', '(none given)')}"
        )
    if target not in known_objects:
        raise PerceptionError(
            f"Gemini returned '{target}', which isn't in known_objects={known_objects}"
        )
    if result.get("visible") is False:
        raise PerceptionError(f"Gemini says '{target}' isn't visible in this frame.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gemini perception on a single saved image.")
    parser.add_argument("--image", required=True, help="Path to a saved face-camera PNG (e.g. scene.png)")
    parser.add_argument("--instruction", required=True, help='e.g. "pick up the green ball"')
    parser.add_argument("--known-objects", nargs="+", default=["green_ball", "red_ball"])
    parser.add_argument("--api-key", default=None, help="Overrides GEMINI_API_KEY env var")
    parser.add_argument(
        "--mode", choices=["target", "action"], default="target",
        help="'target': which object does the instruction mean (query_target_object). "
             "'action': what should the robot do next from this frame (decide_next_action).",
    )
    args = parser.parse_args()

    try:
        if args.mode == "target":
            result = query_target_object(args.image, args.instruction, args.known_objects, api_key=args.api_key)
        else:
            result = decide_next_action(args.image, args.instruction, args.known_objects, api_key=args.api_key)
        print("Perception result:")
        print(json.dumps(result, indent=2))
    except PerceptionError as e:
        print(f"Perception failed: {e}")