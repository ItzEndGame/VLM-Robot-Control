# VLM-Robot-Control

A PyBullet simulation of a two-wheeled humanoid mobile manipulator (twin-wheel base, rising neck column, torso, head-mounted camera, and a functional Franka Panda arm) that uses a Gemini vision-language model as its perception/decision layer to find and pick up a specified object on a table.

Built for a robotics/AI course assignment: the robot's face camera captures a frame, Gemini decides the next discrete action (turn/move/target reached) in a closed-loop visual search, and once the target is close enough the arm executes a physics-grounded pick (and optional place into a colored box).

## How it works

- **`sim_starter.py`** — builds the scene (table + colored balls, optional boxes), builds the kinematic robot body, mounts the Panda arm, runs the closed-loop search/approach/pick/place logic, and drives the PyBullet GUI.
- **`gemini_perception.py`** — standalone perception module. Given a saved camera frame + an instruction + the list of known object names, it asks Gemini either "which object does this instruction mean" (`query_target_object`) or "what's the next navigation action from this frame" (`decide_next_action`). Can be tested independently of PyBullet.

The robot's own safety box and real object poses (from the physics engine) always have the final say — Gemini only decides *which* object and roughly *when* the robot looks close enough; it never controls the arm directly or supplies coordinates.

## Requirements

- Python **3.11** (pinned — PyBullet wheels and this project are validated against 3.11)
- A display (the sim runs `p.connect(p.GUI)`, so it needs a windowing environment; on a headless server you'd need something like a virtual framebuffer)
- A [Gemini API key](https://aistudio.google.com/apikey) — only required for the VLM-driven search; skip it entirely with `--no-perception`

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ItzEndGame/VLM-Robot-Control.git
cd VLM-Robot-Control
```

### 2. Create and activate a Python 3.11 virtual environment

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

> If `python3.11` / `py -3.11` isn't found, install Python 3.11 first (e.g. from [python.org](https://www.python.org/downloads/) or `pyenv install 3.11`).

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install pybullet numpy google-genai pillow python-dotenv
```

- `pybullet`, `numpy` — required for the simulation itself
- `google-genai`, `pillow` — only needed for the Gemini-driven search; safe to skip if you'll only run with `--no-perception`
- `python-dotenv` — optional, lets both scripts auto-load a `.env` file from the working directory

### 4. Configure your Gemini API key

Copy the template and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_api_key_here
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `.env` is loaded automatically by both scripts at startup (via `python-dotenv`) if present — no manual `export`/`setx` needed.

## Running

Run the full simulation, with Gemini-driven visual search:

```bash
python sim_starter.py
```

Run without any Gemini calls (fixed approach point + first known ball — useful for offline testing or if you don't have an API key):

```bash
python sim_starter.py --no-perception
```

Test the perception module on its own, against a saved camera frame:

```bash
python gemini_perception.py --image scene.png --instruction "pick up the green ball"
```

### Giving a natural-language instruction

Pass the command as a single string via `--instruction`:

```bash
python sim_starter.py --instruction "pick up the blue ball and place it in the cyan box"
```

The robot's search target and its pick/place behavior are both parsed straight out of this string — no other flags are needed for a normal run:

- **Which ball** — the color word in the instruction is matched against the balls actually on the table (`red`, `green`, `blue`, `yellow`, `purple`). If no known color is found, the robot searches for any ball instead.
- **What to do after picking** — the instruction is scanned for keywords:
  - mentions `box` / `container` / `bin` / `crate` → the object is placed in a box (a set of colored boxes is added to the far end of the table automatically for this run)
  - mentions `other end` / `far end` / `opposite side` / `across the table` → the object is set down on the table's far edge instead
  - neither → the robot just picks the object up and holds it
- **Which box** — if placing in a box, the box color word in the instruction is matched against the available box colors: `orange`, `cyan`, `magenta`, `white`, `black`. If no box color is named, the robot searches for any box.

More examples:

```bash
# pick only, no place destination
python sim_starter.py --instruction "pick up the red ball"

# place on the far/opposite side of the table instead of a box
python sim_starter.py --instruction "grab the yellow ball and put it on the other end of the table"

# a specific box color
python sim_starter.py --instruction "pick the purple ball and drop it in the white box"
```

> Box colors are `orange`, `cyan`, `magenta`, `white`, `black` (not `red` — that's a ball color). If you ask for a box color that isn't one of these, the robot falls back to searching for any box.

If you want to force the pick/place behavior instead of letting it infer from the instruction text, use `--place {auto,none,other-end,box}` (default `auto`).

## Notes

- The base moves **kinematically** (position is set directly each frame), not via simulated wheel-torque balancing — real self-balancing dynamics were deliberately out of scope for this assignment.
- Only the right arm is a real, IK-driven Panda arm; the left arm is a non-articulated decorative visual match to the reference robot design.
- See the module docstrings in `sim_starter.py` and `gemini_perception.py` for detailed design notes and the reasoning behind specific implementation choices (staged IK waypoints, gradual gripper closing, etc.).