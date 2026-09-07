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
import random
import time

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current working directory (if present)
except ImportError:
    pass  # python-dotenv not installed — GEMINI_API_KEY must be set some other way


SEARCH_ACTIONS = ["turn_left", "turn_right", "move_forward", "move_backward", "target_reached"]


class PerceptionError(Exception):
    """Raised when Gemini perception fails or gives an unusable answer.
    This is the base/transient case — see PerceptionFatalError below for
    the subset that's pointless to retry."""


class PerceptionFatalError(PerceptionError):
    """
    A PerceptionError that will fail EXACTLY the same way on every call,
    no matter how many times it's retried or from what robot pose —
    missing API key, missing package, a bad image path. Retrying these
    (whether by turning, waiting, or anything else) just burns through
    the search step budget for no benefit; the caller should stop and
    report the real problem immediately instead of eventually reporting
    a misleading "target not found".
    """


def _generate_content_with_retry(client, model, contents, config, max_retries=5, base_delay=2.0):
    """
    Call client.models.generate_content, retrying with exponential backoff
    + jitter specifically on 429 (rate limit) errors — the free tier's RPM
    cap is easy to hit in a tight search loop with no built-in pacing
    (verified against a real 429 response: gemini-3.6-flash's free tier
    is 5 requests/minute, tighter than it might look at a glance, and
    this script fires one request per search step with no delay between
    them — a robot that has to turn several times before the target
    comes into view can burn through that in under a minute).

    Any OTHER error (bad key, network failure, etc.) is NOT retried — it's
    re-raised immediately, since retrying those just wastes time before
    failing the same way anyway. Only 429/RESOURCE_EXHAUSTED gets the
    backoff treatment.

    The 429 response body usually includes the server's own suggested
    wait time (a RetryInfo.retryDelay, e.g. "28s") — that's a better
    signal than a blind exponential guess, since it reflects the actual
    remaining time left in the quota window rather than an arbitrary
    schedule. Use it when present; fall back to exponential backoff
    otherwise.
    """
    from google.genai import errors as genai_errors

    def _server_retry_delay(e):
        """Pull RetryInfo.retryDelay (e.g. '28s') out of a 429's details, if present."""
        try:
            for detail in (getattr(e, "details", None) or {}).get("error", {}).get("details", []):
                if detail.get("@type", "").endswith("RetryInfo"):
                    return float(str(detail["retryDelay"]).rstrip("s"))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
        return None

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except genai_errors.APIError as e:
            is_rate_limit = getattr(e, "code", None) == 429 or getattr(e, "status", None) == "RESOURCE_EXHAUSTED"
            if not is_rate_limit:
                raise  # a different API error — don't retry, let the caller handle it
            last_error = e
            if attempt == max_retries:
                break
            server_delay = _server_retry_delay(e)
            delay = server_delay + random.uniform(0, 1.0) if server_delay is not None \
                else base_delay * (2 ** attempt) + random.uniform(0, 1.0)
            source = "server-suggested" if server_delay is not None else "exponential backoff"
            print(f"  [gemini] rate limited (429), retrying in {delay:.1f}s "
                  f"({source}, attempt {attempt + 1}/{max_retries})...")
            time.sleep(delay)

    raise PerceptionError(
        f"Gemini API rate limit (429) persisted after {max_retries} retries. "
        f"Last error: {last_error}. If this keeps happening, your free-tier RPM cap is "
        f"likely too low for the current --max-search-steps — try reducing it, or wait "
        f"a minute for the rate limit window to reset."
    )


def decide_next_action(image_path, instruction, known_objects, api_key=None, model="gemini-3.6-flash",
                        target_kind="object to pick up"):
    """
    One step of a closed-loop visual search: given the robot's CURRENT
    camera view + instruction, decide the single next discrete action —
    instead of the robot driving to one hardcoded point and hoping the
    target happens to be in frame there.

    target_kind describes what `known_objects` actually are, since this
    same function is reused for two different searches:
      - target_kind="object to pick up" (default): known_objects is a
        list of pickable items (e.g. ball colors). The prompt explicitly
        tells the model to ignore any destination mentioned in the
        instruction (e.g. "...and place it in the orange box") —
        otherwise it can latch onto the destination instead of the
        object actually being searched for in THIS phase.
      - target_kind="colored box": known_objects is a list of box labels
        (e.g. "box_orange"). Here the box genuinely IS the target, so the
        "ignore any box mentioned" guidance would be actively wrong —
        this phase needs the opposite framing.

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
        raise PerceptionFatalError("google-genai is not installed. Run: pip install google-genai") from e
    try:
        from PIL import Image
    except ImportError as e:
        raise PerceptionFatalError("Pillow is not installed. Run: pip install pillow") from e

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PerceptionFatalError(
            "No Gemini API key found. Set the GEMINI_API_KEY environment "
            "variable or pass api_key=... explicitly."
        )
    if not os.path.exists(image_path):
        raise PerceptionError(f"Image not found: {os.path.abspath(image_path)}")
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise PerceptionError(f"Could not open image '{image_path}': {e}") from e

    if target_kind == "colored box":
        target_framing = f"""Known {target_kind} candidates that might be visible in the scene: {known_objects}

Each box is painted a distinct solid color matching its label (e.g. the
box labeled "box_orange" is solid orange) — identify it by that color,
the same way you'd identify a colored ball."""
    else:
        target_framing = f"""Known {target_kind} candidates that might be somewhere in the scene: {known_objects}

The instruction may also mention a DESTINATION (e.g. "...and place it in
the orange box", "...at the other end of the table") — ignore that part
entirely for this decision. Your only job here is finding and confirming
the {target_kind}; a separate step handles placing it afterward. Only ever
set target_object to one of the known candidates above, never to a
destination."""

    prompt = f"""You are the navigation brain of a mobile robot searching for
an object. Its front camera just captured the attached image. Its
instruction is:

    "{instruction}"

{target_framing}

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
        response = _generate_content_with_retry(
            client, model, [prompt, image],
            types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except PerceptionError:
        raise
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
        raise PerceptionFatalError(
            "google-genai is not installed. Run: pip install google-genai"
        ) from e

    try:
        from PIL import Image
    except ImportError as e:
        raise PerceptionFatalError("Pillow is not installed. Run: pip install pillow") from e

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise PerceptionFatalError(
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
        response = _generate_content_with_retry(
            client, model, [prompt, image],
            types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except PerceptionError:
        raise
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