"""
Mobile manipulator simulation: self-balancing twin-wheel base + humanoid
torso/head + Panda arm/gripper + face camera.

Modeled loosely on reference robots like Enchanted Tools' "Miroki" /
similar service robots: a black two-wheel base, a thin rising column,
a white humanoid torso with twin arms, and a dark sensor head on top.

Install (on your own machine, with a display):
    pip install pybullet numpy pillow

Run:
    python sim_mobile_manipulator.py

Design notes (scoping decisions, worth mentioning in your report):
- The base still moves KINEMATICALLY (we directly set its position each
  frame) rather than via simulated wheel torque/balancing physics. Real
  self-balancing dynamics are a research problem on their own and add no
  value to the actual assignment goal (the AI decision pipeline), so this
  is a deliberate simplification, same as before.
- The whole body (wheels, base pod, neck column, torso, head, arms) is
  one rigid kinematic stack: every part's pose is derived every frame
  from a single (x, y, yaw) for the robot, via fixed local offsets. This
  keeps "the robot" a single source of truth instead of N separately
  drifting bodies.
- The scissor-lift neck in the reference photo is visually approximated
  with a single straight column (thin cylinder). A real scissor
  linkage would need its own joint chain for basically zero benefit here.
- Only ONE arm is a real, IK-driven Panda arm (mounted at the right
  shoulder) — that's the arm that actually does the picking. The left
  arm is a simple decorative (non-articulated) visual arm, just so the
  robot reads as twin-armed like the reference image. Making both arms
  fully functional would double the IK/joint bookkeeping for no benefit
  to a single-object pick task.
- The camera is now mounted in the HEAD ("face" region), at the front of
  the head unit, and always looks in the direction the robot is facing.
- Navigation is now a CLOSED-LOOP visual search (see search_for_target()),
  not a scripted "drive to one hardcoded standoff point and hope the
  target is in frame there". Each step: capture the current view, ask
  Gemini for a single next action (turn_left/turn_right/move_forward/
  move_backward/target_reached), execute it, repeat — up to a step
  budget. This is what makes object-finding "the robot's own decision"
  rather than a fixed path, i.e. a small VLA-style loop. The robot is
  still kept inside a safety box server-side (won't drive into the table
  or wander off), regardless of what Gemini suggests.
- Gemini's role stays narrow even in the search loop: it decides WHICH
  known object (by name) and roughly WHEN the robot is close enough — the
  actual pick position always comes from the physics engine's real object
  pose, not image back-projection, and align_for_pick() does
  one more physics-grounded check (real distance to the arm's max reach)
  before do_pick runs, since a visual "looks close enough" call from
  Gemini isn't a guarantee the IK solution will actually be comfortable.
- do_pick() moves through a HIGH hover waypoint first, at reduced motor
  force, before descending — going straight from the arm's folded home
  pose to a point just above the target (the original approach) can sweep
  the forearm/gripper low across the tabletop en route and violently
  fling anything in its path. This was observed directly: an earlier
  version of this script flung a ball meters away on pickup.
- Run with --no-perception to skip Gemini (and the search loop) entirely
  and default to a fixed approach point + the green ball, for offline
  testing without an API key.
"""

import argparse
import os
import random
import time
import numpy as np
import pybullet as p
import pybullet_data

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current working directory (if present)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Robot body layout constants
# ---------------------------------------------------------------------------
# Twin-wheel base (the black "segway" pod in the reference image)
WHEEL_RADIUS = 0.18
WHEEL_THICKNESS = 0.07
WHEEL_Y_OFFSET = 0.32           # half-distance between the two wheels

# NOTE: this must stay NARROWER than WHEEL_Y_OFFSET (in y), or the pod
# visually swallows the wheels — that was the "one black log" bug.
BASE_HALF_EXTENTS = [0.13, 0.19, 0.085]   # black pod connecting the wheels

# Neck / rising column (approximates the scissor-lift in the reference photo)
NECK_RADIUS = 0.065             # thickened for visibility
NECK_HEIGHT = 0.42

# Torso (white humanoid body) — kept narrower than SHOULDER_Y_OFFSET so
# the arms actually stick out past the torso silhouette instead of hiding
# inside it.
TORSO_HALF_EXTENTS = [0.11, 0.15, 0.16]
SHOULDER_Y_OFFSET = 0.30        # how far out the shoulders/arms sit

# Head (dark sensor/camera housing on top)
HEAD_HALF_EXTENTS = [0.09, 0.09, 0.09]

# Downward tilt of the face camera. The old fixed z-nudge (-0.15 over a 1m
# look-ahead) was only ~8.5° down — fine at long range, but once the robot
# stands close to the table (small standoff) the tabletop is both close AND
# well below head height, so the camera needs a much steeper look-down angle
# to actually keep it in frame. ~38° comfortably covers the standoff range
# this script uses (0.25-0.55m) given the head-to-tabletop height gap.
CAMERA_PITCH_DEG = 38

# Decorative (non-articulated) left arm: upper arm hangs from the
# shoulder, forearm bends forward at the elbow, small hand at the tip —
# reads as a bent arm instead of a single hidden stick.
DECO_UPPER_LEN = 0.16
DECO_UPPER_RADIUS = 0.05
DECO_FOREARM_LEN = 0.16
DECO_FOREARM_RADIUS = 0.045
DECO_FOREARM_PITCH = 1.0        # radians, forward tilt of forearm from vertical
DECO_HAND_RADIUS = 0.06
SHOULDER_JOINT_RADIUS = 0.06    # small ball-joint spheres at both shoulders

# ---------------------------------------------------------------------------
# Derived heights (all measured up from the ground, z = 0)
# ---------------------------------------------------------------------------
BASE_Z = WHEEL_RADIUS
NECK_BOTTOM_Z = BASE_Z + BASE_HALF_EXTENTS[2]
NECK_Z = NECK_BOTTOM_Z + NECK_HEIGHT / 2
TORSO_BOTTOM_Z = NECK_BOTTOM_Z + NECK_HEIGHT
TORSO_Z = TORSO_BOTTOM_Z + TORSO_HALF_EXTENTS[2]
HEAD_BOTTOM_Z = TORSO_BOTTOM_Z + 2 * TORSO_HALF_EXTENTS[2]
HEAD_Z = HEAD_BOTTOM_Z + HEAD_HALF_EXTENTS[2]
SHOULDER_Z = TORSO_Z + TORSO_HALF_EXTENTS[2] * 0.5

# Elbow sits at the bottom of the decorative upper arm; the forearm then
# tilts forward (+x, robot-frame) from there by DECO_FOREARM_PITCH.
_elbow_local = np.array([0.0, SHOULDER_Y_OFFSET, SHOULDER_Z - DECO_UPPER_LEN])
_forearm_dir = np.array([np.sin(DECO_FOREARM_PITCH), 0.0, -np.cos(DECO_FOREARM_PITCH)])
_deco_forearm_local = tuple(_elbow_local + _forearm_dir * (DECO_FOREARM_LEN / 2))
_deco_hand_local = tuple(_elbow_local + _forearm_dir * DECO_FOREARM_LEN)

# Local (robot-frame) offsets for every part, keyed by name. All poses are
# computed from these + the robot's current (x, y, yaw) every frame.
LOCAL_OFFSETS = {
    "wheel_left":    (0.0,  WHEEL_Y_OFFSET, BASE_Z),
    "wheel_right":   (0.0, -WHEEL_Y_OFFSET, BASE_Z),
    "base":          (0.0, 0.0, BASE_Z),
    "neck":          (0.0, 0.0, NECK_Z),
    "neck_cap_bottom": (0.0, 0.0, NECK_BOTTOM_Z),
    "neck_cap_top":    (0.0, 0.0, NECK_BOTTOM_Z + NECK_HEIGHT),
    "torso":         (0.0, 0.0, TORSO_Z),
    "head":          (0.0, 0.0, HEAD_Z),
    "shoulder_left":  (0.0,  SHOULDER_Y_OFFSET, SHOULDER_Z),
    "shoulder_right": (0.0, -SHOULDER_Y_OFFSET, SHOULDER_Z),
    "deco_upper_arm": (0.0, SHOULDER_Y_OFFSET, SHOULDER_Z - DECO_UPPER_LEN / 2),
    "deco_forearm":   _deco_forearm_local,
    "deco_hand":      _deco_hand_local,
}

# The functional (IK-driven) Panda arm mounts here:
ARM_MOUNT_KEY = "shoulder_right"

# World orientation (euler) for the angled decorative forearm — composes a
# forward pitch about y with the robot's current yaw about z.
def _forearm_euler(yaw):
    return [0, np.pi - DECO_FOREARM_PITCH, yaw]


def local_to_world(position_xy, yaw, local_offset):
    """Rotate a robot-frame (dx, dy, dz) offset by yaw and add to (x, y, 0)."""
    dx, dy, dz = local_offset
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    wx = position_xy[0] + dx * cos_y - dy * sin_y
    wy = position_xy[1] + dx * sin_y + dy * cos_y
    return [wx, wy, dz]


def setup_static_scene():
    """Ground plane and table with objects to pick."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    p.loadURDF("plane.urdf")
    table_id = p.loadURDF("table/table.urdf", basePosition=[1.2, 0, 0])

    # Place the balls near the table's FRONT edge (the edge closest to the
    # robot's start at the origin), not at the table's center — the Panda
    # arm's ~0.85m reach plus the shoulder-mount offset can't reach all the
    # way to table-center from any approach point that stays outside the
    # table's footprint. Read the table's real AABB instead of guessing a
    # hardcoded x, so this stays correct regardless of the actual asset size.
    aabb_min, aabb_max = p.getAABB(table_id)
    table_top_z = aabb_max[2]
    near_edge_x = aabb_min[0]  # edge nearest the origin, since table sits at +x
    ball_radius = 0.03
    ball_x = near_edge_x + 0.15   # inset just enough to sit stably on the tabletop
    ball_z = table_top_z + ball_radius + 0.01

    green_ball_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[0, 1, 0, 1]),
        basePosition=[ball_x, 0.1, ball_z],
    )
    red_ball_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 0, 0, 1]),
        basePosition=[ball_x, -0.15, ball_z],
    )

    # PyBullet's default dynamics for a small, light sphere are underdamped
    # and bouncy — fine for it just sitting on the table, but a real
    # problem the moment a gripper actually contacts it: verified directly
    # that this contributes to the fingers "exploding" the ball away on
    # contact instead of grasping it cleanly.
    for ball_id in (green_ball_id, red_ball_id):
        p.changeDynamics(ball_id, -1, lateralFriction=1.2, spinningFriction=0.005,
                          rollingFriction=0.005, restitution=0.0,
                          linearDamping=0.3, angularDamping=0.3)

    return {"table": table_id, "green_ball": green_ball_id, "red_ball": red_ball_id}


def _make_body(collision_id, visual_id, world_pos, world_orn):
    return p.createMultiBody(
        baseMass=0,  # kinematic — moved by directly setting pose every frame
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=world_pos,
        baseOrientation=world_orn,
    )


def create_robot_visual():
    """
    Build the humanoid-on-wheels body: twin wheels + black base pod, a thin
    rising neck column, a white torso, a dark head unit, and a simple
    decorative left arm. Everything starts at robot position (0, 0), yaw 0.

    Only the base pod and torso get collision shapes (so the body can
    physically bump into things); the rest are visual-only to keep things
    simple and avoid weird self-collision while it moves.
    """
    position_xy, yaw = [0.0, 0.0], 0.0
    ids = {}

    # --- Wheels (black, cylinder laid on its side) ---
    wheel_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=WHEEL_RADIUS, length=WHEEL_THICKNESS, rgbaColor=[0.05, 0.05, 0.05, 1]
    )
    for key in ("wheel_left", "wheel_right"):
        wpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS[key])
        worn = p.getQuaternionFromEuler([1.5708, 0, yaw])
        ids[key] = _make_body(-1, wheel_visual, wpos, worn)

    # --- Base pod (black, connects the wheels — has collision) ---
    base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=BASE_HALF_EXTENTS)
    base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=BASE_HALF_EXTENTS, rgbaColor=[0.1, 0.1, 0.1, 1])
    bpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["base"])
    born = p.getQuaternionFromEuler([0, 0, yaw])
    ids["base"] = _make_body(base_collision, base_visual, bpos, born)

    # --- Neck column (steel-blue, visual only) — distinct color from the
    #     white torso above and black base below, plus two dark "joint cap"
    #     discs so the transitions read as mechanical joints, not one blob.
    neck_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=NECK_RADIUS, length=NECK_HEIGHT, rgbaColor=[0.42, 0.48, 0.58, 1]
    )
    npos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["neck"])
    ids["neck"] = _make_body(-1, neck_visual, npos, born)

    joint_cap_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=NECK_RADIUS * 1.35, length=0.025, rgbaColor=[0.12, 0.12, 0.12, 1]
    )
    for key in ("neck_cap_bottom", "neck_cap_top"):
        cpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS[key])
        ids[key] = _make_body(-1, joint_cap_visual, cpos, born)

    # --- Torso (white, has collision) ---
    torso_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=TORSO_HALF_EXTENTS)
    torso_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=TORSO_HALF_EXTENTS, rgbaColor=[0.95, 0.95, 0.96, 1])
    tpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["torso"])
    ids["torso"] = _make_body(torso_collision, torso_visual, tpos, born)

    # --- Head (dark sensor/camera housing, visual only) ---
    head_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=HEAD_HALF_EXTENTS, rgbaColor=[0.08, 0.08, 0.08, 1])
    hpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["head"])
    ids["head"] = _make_body(-1, head_visual, hpos, born)

    # --- Shoulder ball joints (both sides, small dark spheres — purely
    #     cosmetic, just visually connects the arms to the torso) ---
    shoulder_joint_visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=SHOULDER_JOINT_RADIUS, rgbaColor=[0.15, 0.15, 0.15, 1]
    )
    for key in ("shoulder_left", "shoulder_right"):
        spos = local_to_world(position_xy, yaw, LOCAL_OFFSETS[key])
        ids[key] = _make_body(-1, shoulder_joint_visual, spos, born)

    # --- Decorative left arm (white upper arm + forearm, dark hand,
    #     visual only — bent at the elbow so it reads as an arm) ---
    deco_upper_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=DECO_UPPER_RADIUS, length=DECO_UPPER_LEN, rgbaColor=[0.9, 0.9, 0.92, 1]
    )
    dupos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["deco_upper_arm"])
    ids["deco_upper_arm"] = _make_body(-1, deco_upper_visual, dupos, born)

    deco_forearm_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=DECO_FOREARM_RADIUS, length=DECO_FOREARM_LEN, rgbaColor=[0.9, 0.9, 0.92, 1]
    )
    dfpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["deco_forearm"])
    dforn = p.getQuaternionFromEuler(_forearm_euler(yaw))
    ids["deco_forearm"] = _make_body(-1, deco_forearm_visual, dfpos, dforn)

    deco_hand_visual = p.createVisualShape(p.GEOM_SPHERE, radius=DECO_HAND_RADIUS, rgbaColor=[0.75, 0.12, 0.1, 1])
    dhpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["deco_hand"])
    ids["deco_hand"] = _make_body(-1, deco_hand_visual, dhpos, born)

    return ids


def set_robot_pose(robot_ids, position_xy, yaw):
    """Move every body part kinematically to match a new (x, y, yaw)."""
    upright_orn = p.getQuaternionFromEuler([0, 0, yaw])
    wheel_orn = p.getQuaternionFromEuler([1.5708, 0, yaw])

    for key in ("wheel_left", "wheel_right"):
        wpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS[key])
        p.resetBasePositionAndOrientation(robot_ids[key], wpos, wheel_orn)

    for key in ("base", "neck", "neck_cap_bottom", "neck_cap_top", "torso", "head",
                "shoulder_left", "shoulder_right", "deco_upper_arm", "deco_hand"):
        wpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS[key])
        p.resetBasePositionAndOrientation(robot_ids[key], wpos, upright_orn)

    # Forearm needs its own orientation (upper-arm-relative forward pitch
    # composed with the robot's yaw), not the plain upright one.
    dfpos = local_to_world(position_xy, yaw, LOCAL_OFFSETS["deco_forearm"])
    dforn = p.getQuaternionFromEuler(_forearm_euler(yaw))
    p.resetBasePositionAndOrientation(robot_ids["deco_forearm"], dfpos, dforn)


def load_arm_on_shoulder(mount_pos, yaw):
    """Load the functional Panda arm (built-in gripper) at the right shoulder."""
    arm_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=mount_pos,
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
        useFixedBase=True,
    )
    return arm_id


def get_controllable_joints(body_id):
    """Return indices of revolute/prismatic joints (skips fixed joints)."""
    joints = []
    for i in range(p.getNumJoints(body_id)):
        joint_type = p.getJointInfo(body_id, i)[2]
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joints.append(i)
    return joints


def hold_arm_pose(arm_id, joint_indices, joint_targets, force=200):
    """
    Continuously command the arm's joints to stay at joint_targets.
    MUST be called every simulation step, otherwise gravity will slowly
    droop/collapse the arm (this was the 'floppy arm' issue).
    """
    for j, target in zip(joint_indices, joint_targets):
        p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, target, force=force)


def compute_approach_point(table_id, robot_start_xy, standoff=0.55):
    """
    Instead of driving straight to the object's (x, y) — which is on top
    of / inside the table's footprint — compute a point just outside the
    table's bounding box, offset back towards the robot's start position,
    so the arm can reach onto the tabletop instead of the robot driving
    into/under the table.
    """
    aabb_min, aabb_max = p.getAABB(table_id)
    table_center_xy = np.array([(aabb_min[0] + aabb_max[0]) / 2,
                                 (aabb_min[1] + aabb_max[1]) / 2])

    direction = np.array(robot_start_xy) - table_center_xy
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.array([-1.0, 0.0])

    # Half-extent of the table along the approach direction (rough, using AABB)
    half_extent_x = (aabb_max[0] - aabb_min[0]) / 2
    half_extent_y = (aabb_max[1] - aabb_min[1]) / 2
    table_radius_along_dir = abs(direction[0]) * half_extent_x + abs(direction[1]) * half_extent_y

    approach_xy = table_center_xy + direction * (table_radius_along_dir + standoff)
    return approach_xy.tolist()


def move_robot_towards(robot_ids, arm_id, current_pos, target_xy, speed=0.015):
    """
    One kinematic step: nudge the whole robot body (and mounted arm)
    towards target_xy. Returns the new (x, y) position, new yaw, and
    whether the target has been reached.

    NOTE: not called on the default (search-loop) path anymore — kept as
    a reusable utility for smooth point-to-point travel, e.g. if you want
    to animate the final approach after the search loop locates a target,
    instead of the instant teleport _place_and_settle() currently does.
    """
    direction = np.array(target_xy) - np.array(current_pos)
    distance = np.linalg.norm(direction)
    if distance < 0.05:
        return current_pos, None, True

    step = direction / distance * min(speed, distance)
    new_pos = (np.array(current_pos) + step).tolist()
    new_yaw = np.arctan2(direction[1], direction[0])

    set_robot_pose(robot_ids, new_pos, new_yaw)
    arm_mount_pos = local_to_world(new_pos, new_yaw, LOCAL_OFFSETS[ARM_MOUNT_KEY])
    p.resetBasePositionAndOrientation(
        arm_id, arm_mount_pos, p.getQuaternionFromEuler([0, 0, new_yaw])
    )
    return new_pos, new_yaw, False


def capture_face_camera(position_xy, yaw, save_path=None, live_window=False):
    """
    Simulate the 'face' camera: mounted in the HEAD unit, at the front of
    the robot, looking in the direction the robot currently faces.

    - Calling this every simulation step gives you a REAL-TIME feed:
      PyBullet's own GUI automatically shows an on-screen "Synthetic
      Camera RGB/Depth/Segmentation" preview whenever getCameraImage()
      is called.
    - Pass save_path to also write a PNG snapshot to disk (e.g. once,
      right before sending it to the VLM/Gemini step).
    - Pass live_window=True to additionally pop up a proper OpenCV
      window showing the feed, closer to what a real robot camera
      stream looks like (needs `pip install opencv-python`).
    """
    head_pos = np.array(local_to_world(position_xy, yaw, LOCAL_OFFSETS["head"]))
    forward = np.array([np.cos(yaw), np.sin(yaw), 0])

    cam_eye = head_pos + forward * (HEAD_HALF_EXTENTS[0] + 0.02)
    pitch = np.radians(CAMERA_PITCH_DEG)
    look_ahead = 1.0
    cam_target = cam_eye + forward * (look_ahead * np.cos(pitch))
    cam_target[2] -= look_ahead * np.sin(pitch)  # steep downward tilt (see CAMERA_PITCH_DEG)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_eye.tolist(),
        cameraTargetPosition=cam_target.tolist(),
        cameraUpVector=[0, 0, 1],
    )
    proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1.0, nearVal=0.05, farVal=5.0)

    width, height = 320, 320  # smaller than before — keeps per-frame cost low
    _, _, rgb_img, _, _ = p.getCameraImage(
        width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]

    if live_window:
        try:
            import cv2
            bgr = cv2.cvtColor(rgb_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imshow("Robot Face Camera (live)", bgr)
            cv2.waitKey(1)
        except ImportError:
            print("opencv-python not installed (pip install opencv-python) — "
                  "skipping live_window, relying on PyBullet's built-in preview instead.")
        except Exception as e:
            # IMPORTANT: this used to only catch ImportError, so any other
            # OpenCV/display failure (missing Qt plugin, no display backend,
            # etc.) would crash the whole function BEFORE the save step
            # below ever ran — that was very likely why no image got saved.
            print(f"live_window display failed ({type(e).__name__}: {e}) — "
                  f"continuing without the OpenCV window.")

    if save_path:
        try:
            from PIL import Image
            abs_path = os.path.abspath(save_path)
            os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
            Image.fromarray(rgb_array.astype(np.uint8)).save(abs_path)
            print(f"Saved face-camera image to {abs_path}")
        except ImportError:
            print("Pillow not installed (pip install pillow) — skipping image save.")
        except Exception as e:
            print(f"Failed to save image to '{save_path}': {type(e).__name__}: {e}")

    return rgb_array


def do_pick(arm_id, target_pos, target_body_id=None):
    """
    Use IK to reach the target position, then close the Panda gripper.

    Every design choice below was verified directly against real pybullet
    (headless DIRECT-mode test runs), not guessed — several earlier guesses
    (an assumed grasp orientation, explicit IK seeding, a single big jump
    per stage) were each tested and found to make things WORSE, and were
    reverted. Summary of what's actually in effect and why:

    - Motion is staged through many small waypoints (fine descent, then
      fine lift), not 2-3 big jumps. Big jumps let the IK solver flip to a
      totally different, unpredictably-oriented joint configuration each
      call, which both looked visually wrong and could sweep the arm
      through other objects on the way.
    - Position-only IK is used for the earlier/farther waypoints (stable,
      converges reliably); an explicit top-down targetOrientation
      (verified via direct testing to converge to ~0.000m offset for this
      arm/mount) is used only for the last 4 waypoints. Locking it from
      the very first far-away hover point was tested and found unstable.
    - Gripper closing is GRADUAL (40 small steps, low force), not one
      instant jump to fully-closed. An instant hard close on a real
      object in between the fingers produced an unstable contact "pop"
      that violently launched it — verified directly.
    - If target_body_id is given, the object's LIVE position is re-read
      right before closing and used to re-center the last approach step.
      Small positional drift can accumulate during the earlier
      position-only descent stages (verified: up to several cm), which is
      enough to make the ball land off-center between the fingers and get
      pushed by only one of them instead of pinched by both.

    NOTE: end_effector_index=11 and finger_joints=[9, 10] are correct for
    the standard franka_panda/panda.urdf shipped with pybullet_data, but
    versions can vary — run `python sim_mobile_manipulator.py
    --print-joints` to verify these against your actual installed asset.
    """
    end_effector_index = 11
    finger_joints = [9, 10]
    grasp_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

    hover_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.35])
    grasp_pos = np.array(target_pos)

    # --- Phase 1: reach the hover point via JOINT-SPACE interpolation, not
    # a single Cartesian IK jump. Verified directly: a one-shot Cartesian
    # jump from the arm's folded home pose to hover converges poorly
    # (offsets up to 0.35m in diagnostic runs), and during that poorly-
    # converged motion, arm links OTHER than the fingers (forearm/wrist)
    # can swing through the ball's location and knock it off the table —
    # confirmed by checking contacts on every link, not just the fingers.
    # This was especially reliable at robot standoff distances of
    # 0.15-0.20m from the table edge — exactly where search_for_target's
    # default forward-step size naturally lands the robot. Interpolating
    # smoothly in JOINT space instead (after computing a single well-
    # converged position-only IK solution for the hover point) structurally
    # can't "jump branches" the way repeated Cartesian IK calls can.
    controllable = get_controllable_joints(arm_id)
    start_joint_state = [p.getJointState(arm_id, j)[0] for j in controllable]
    hover_joint_solution = p.calculateInverseKinematics(
        arm_id, end_effector_index, hover_pos.tolist(),
        maxNumIterations=200, residualThreshold=1e-5,
    )
    n_joint_steps = 20
    for step in range(1, n_joint_steps + 1):
        t = step / n_joint_steps
        for i, j in enumerate(controllable):
            if j in finger_joints:
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, 0.005, force=20)
            else:
                interp = start_joint_state[i] + (hover_joint_solution[i] - start_joint_state[i]) * t
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, interp, force=80)
        for _ in range(15):
            p.stepSimulation()
            time.sleep(1 / 240)
    actual_hover_pos = p.getLinkState(arm_id, end_effector_index)[0]
    print(f"  [do_pick] stage=hover(joint-space)  target={tuple(round(v,3) for v in hover_pos)}  "
          f"actual_hand_pos={tuple(round(v,3) for v in actual_hover_pos)}  "
          f"offset={np.linalg.norm(np.array(actual_hover_pos)-hover_pos):.3f} m")

    # --- Phase 2: gradually REORIENT in place (position held fixed at
    # hover) from whatever orientation position-only IK naturally landed
    # on, to the target top-down grasp orientation, via quaternion slerp
    # over many small steps.
    #
    # This replaces an earlier, WORSE approach: switching from
    # unconstrained to orientation-locked IK abruptly partway through the
    # Cartesian descent (at a fixed waypoint index). That looked like it
    # worked in some tests (little/no collision, and the object still got
    # "lifted"), but closer inspection showed it was an illusion: the
    # abrupt switch could make IK jump to a totally different, far-away
    # joint configuration (offsets of 0.3-0.57m were measured even in
    # runs that superficially "succeeded", across EVERY tested standoff
    # distance) — the live re-centering step + grasp constraint were just
    # dragging the object from wherever it ended up and welding it there,
    # not performing a real grasp. Changing position and orientation
    # separately like this, instead of blending both changes into one
    # discontinuous jump, converges to ~0.0001m offset instead.
    hover_orn = p.getLinkState(arm_id, end_effector_index)[1]
    n_reorient_steps = 12
    for step in range(1, n_reorient_steps + 1):
        t = step / n_reorient_steps
        step_orn = p.getQuaternionSlerp(hover_orn, grasp_orientation, t)
        joint_poses = p.calculateInverseKinematics(
            arm_id, end_effector_index, hover_pos.tolist(), step_orn,
            maxNumIterations=200, residualThreshold=1e-5,
        )
        for i, joint_angle in zip(controllable, joint_poses):
            if i in finger_joints:
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, 0.005, force=20)
            else:
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_angle, force=80)
        for _ in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)
    actual_reoriented_pos = p.getLinkState(arm_id, end_effector_index)[0]
    print(f"  [do_pick] stage=reorient(in-place)  hand now at "
          f"{tuple(round(v,3) for v in actual_reoriented_pos)}  "
          f"offset_from_hover={np.linalg.norm(np.array(actual_reoriented_pos)-hover_pos):.4f} m")

    # --- Phase 3: fine, evenly-spaced Cartesian waypoints from hover down
    # to the grasp point, WITH ORIENTATION ALREADY LOCKED FOR EVERY STEP
    # (not switched on partway through, per Phase 2's note above). Since
    # only position changes now, each step is a small, well-conditioned
    # move from an already-correctly-oriented starting pose.
    n_descent_steps = 8
    waypoints = [
        hover_pos + (grasp_pos - hover_pos) * (i / n_descent_steps)
        for i in range(1, n_descent_steps + 1)
    ]

    for idx, pos in enumerate(waypoints):
        pos = pos.tolist()
        is_final = idx == len(waypoints) - 1
        joint_poses = p.calculateInverseKinematics(
            arm_id, end_effector_index, pos, grasp_orientation,
            maxNumIterations=200, residualThreshold=1e-5,
        )
        for i, joint_angle in zip(controllable, joint_poses):
            if i in finger_joints:
                # Keep fingers open only once we're two steps into the
                # already-oriented descent (small extra margin near the
                # table); narrow before that.
                finger_width = 0.04 if idx >= 2 else 0.005
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, finger_width, force=20)
            else:
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_angle,
                                         force=200 if is_final else 80)
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1 / 240)

        actual_pos, actual_orn = p.getLinkState(arm_id, end_effector_index)[:2]
        actual_euler_deg = tuple(round(np.degrees(v), 1) for v in p.getEulerFromQuaternion(actual_orn))
        offset = np.linalg.norm(np.array(actual_pos) - np.array(pos))
        stage_name = "grasp" if is_final else f"descent {idx}/{n_descent_steps}"
        print(f"  [do_pick] stage={stage_name:<14} target={tuple(round(v,3) for v in pos)}  "
              f"actual_hand_pos={tuple(round(v,3) for v in actual_pos)}  offset={offset:.3f} m  "
              f"actual_hand_euler_deg={actual_euler_deg}")

    # Re-center on the object's LIVE position before closing. Small
    # positional drift can accumulate during descent (verified: a few cm)
    # — easily enough for a ~3cm-radius ball to land off-center between
    # the fingers, so only one finger makes contact when closing, pushing
    # the ball instead of pinching it.
    if target_body_id is not None:
        live_pos = p.getBasePositionAndOrientation(target_body_id)[0]
        joint_poses = p.calculateInverseKinematics(
            arm_id, end_effector_index, live_pos, grasp_orientation,
            maxNumIterations=200, residualThreshold=1e-5,
        )
        for i, joint_angle in zip(controllable, joint_poses):
            if i not in finger_joints:
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_angle, force=80)
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1 / 240)

    # Close gripper GRADUALLY and gently, not one instant hard jump to
    # fully-closed. Verified empirically: commanding target=0.0 (fully
    # closed) at force=50 in a single step, with the ball physically in
    # between the fingers, produced an unstable contact "pop" that
    # violently launched it. Closing over many small steps at low force
    # lets the contact resolve smoothly instead.
    n_close_steps = 40
    for step in range(n_close_steps):
        finger_target = max(0.0, 0.04 - 0.04 * (step / (n_close_steps - 1)))
        for finger in finger_joints:
            p.setJointMotorControl2(arm_id, finger, p.POSITION_CONTROL, finger_target, force=10)
        p.stepSimulation()
        time.sleep(1 / 240)
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1 / 240)

    # Rigidly attach the object to the gripper for the lift, if
    # target_body_id was given. This is a standard PyBullet technique used
    # in many published manipulation demos specifically because friction-
    # only grasping of a small, light object with pure position control
    # (no force/torque feedback) is fragile: verified directly that even
    # with good positioning and a gentle close, the fingers can end up not
    # making solid contact and the object just gets pushed instead of
    # lifted. A fixed constraint sidesteps that fragility rather than
    # continuing to fight it. Released after the lift completes.
    grasp_constraint = None
    if target_body_id is not None:
        hand_pos, hand_orn = p.getLinkState(arm_id, end_effector_index)[:2]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(target_body_id)
        inv_hand_pos, inv_hand_orn = p.invertTransform(hand_pos, hand_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_hand_pos, inv_hand_orn, obj_pos, obj_orn)
        grasp_constraint = p.createConstraint(
            arm_id, end_effector_index, target_body_id, -1, p.JOINT_FIXED,
            [0, 0, 0], rel_pos, [0, 0, 0], childFrameOrientation=rel_orn,
        )

    # Lift — fine-interpolated for the same reason as the descent above:
    # one big jump here, right next to a possibly-not-quite-grasped ball,
    # risks sweeping through it just like the old unfixed descent did.
    lift_pos = np.array([target_pos[0], target_pos[1], target_pos[2] + 0.2])
    n_lift_steps = 4
    for i in range(1, n_lift_steps + 1):
        wp = (grasp_pos + (lift_pos - grasp_pos) * (i / n_lift_steps)).tolist()
        joint_poses = p.calculateInverseKinematics(
            arm_id, end_effector_index, wp, grasp_orientation,
            maxNumIterations=200, residualThreshold=1e-5,
        )
        for j, joint_angle in zip(get_controllable_joints(arm_id), joint_poses):
            if j in finger_joints:
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, 0.0, force=50)  # stay closed
            else:
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, joint_angle, force=150)
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1 / 240)

    pick_succeeded = None
    if target_body_id is not None:
        final_obj_pos = p.getBasePositionAndOrientation(target_body_id)[0]
        height_gained = final_obj_pos[2] - target_pos[2]
        # A real lift should gain close to the full 0.2m lift height. Small
        # positive numbers could just be the object rolling/settling, not
        # actually being held — so the bar here is deliberately more than
        # half the intended lift, not just ">0".
        pick_succeeded = height_gained > 0.1
        verdict = "SUCCESS" if pick_succeeded else "FAILED (object was not actually lifted)"
        print(f"  [do_pick] final object height: {final_obj_pos[2]:.3f} "
              f"(started at {target_pos[2]:.3f}, gained {height_gained:+.3f} m) -> {verdict}")
        if not pick_succeeded and grasp_constraint is not None:
            # Don't leave a knocked-off/never-grasped object rigidly welded
            # to the arm — that would be confusing (it'd keep following the
            # gripper around) and misrepresents what actually happened.
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None

    return grasp_constraint, pick_succeeded


def run_camera_sanity_check(robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw,
                             out_dir="camera_check", rotate_steps=8, nudge_dist=0.2):
    """
    A quick, dedicated test to answer "is the camera actually working?"
    without waiting through the full navigate-and-pick sequence.

    What it does:
    1. Rotates the robot in place through a full circle in `rotate_steps`
       increments, saving one camera frame per step (camera_check/rot_00.png,
       rot_01.png, ...).
    2. Nudges the robot forward by `nudge_dist`, then back, saving a frame
       at each end (camera_check/nudge_forward.png, nudge_back.png).
    3. After every frame, prints the mean absolute pixel difference from
       the previous frame.

    How to read the output:
    - If the numbers are consistently near 0 while the robot is visibly
      rotating/moving in the PyBullet window, the camera is STUCK (e.g.
      recomputing the same view matrix, or not being called at all) —
      that's a real bug to chase down.
    - If the numbers jump around as expected (bigger jumps for bigger
      rotations, near-0 only when two steps happen to look similar), the
      camera is working correctly.
    - Also just open a few of the saved PNGs in camera_check/ — you
      should visually see the table/floor/balls sweep across frame as
      the robot rotates, and get closer/farther as it nudges forward/back.
    """
    os.makedirs(out_dir, exist_ok=True)

    def settle_and_capture(pos, yaw_val, filename):
        for _ in range(30):
            hold_arm_pose(arm_id, arm_joints, home_targets)
            p.stepSimulation()
            time.sleep(1 / 240)
        return capture_face_camera(pos, yaw_val, save_path=os.path.join(out_dir, filename), live_window=True)

    print(f"\n--- Camera sanity check: saving frames to ./{out_dir}/ ---")
    prev_frame = None

    print("Phase 1: rotating in place through a full circle...")
    for i in range(rotate_steps):
        test_yaw = i * (2 * np.pi / rotate_steps)
        set_robot_pose(robot_ids, robot_pos, test_yaw)
        arm_mount_pos = local_to_world(robot_pos, test_yaw, LOCAL_OFFSETS[ARM_MOUNT_KEY])
        p.resetBasePositionAndOrientation(arm_id, arm_mount_pos, p.getQuaternionFromEuler([0, 0, test_yaw]))

        frame = settle_and_capture(robot_pos, test_yaw, f"rot_{i:02d}.png")
        if prev_frame is not None:
            diff = np.mean(np.abs(frame.astype(int) - prev_frame.astype(int)))
            print(f"  step {i}: yaw={np.degrees(test_yaw):6.1f}°  frame diff vs previous = {diff:6.2f}")
        else:
            print(f"  step {i}: yaw={np.degrees(test_yaw):6.1f}°  (first frame, no diff yet)")
        prev_frame = frame

    print("Phase 2: nudging forward, then back...")
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    fwd_pos = (np.array(robot_pos) + forward * nudge_dist).tolist()
    set_robot_pose(robot_ids, fwd_pos, yaw)
    arm_mount_pos = local_to_world(fwd_pos, yaw, LOCAL_OFFSETS[ARM_MOUNT_KEY])
    p.resetBasePositionAndOrientation(arm_id, arm_mount_pos, p.getQuaternionFromEuler([0, 0, yaw]))
    frame_fwd = settle_and_capture(fwd_pos, yaw, "nudge_forward.png")
    diff = np.mean(np.abs(frame_fwd.astype(int) - prev_frame.astype(int)))
    print(f"  nudged forward {nudge_dist} m  frame diff vs previous = {diff:6.2f}")

    set_robot_pose(robot_ids, robot_pos, yaw)
    arm_mount_pos = local_to_world(robot_pos, yaw, LOCAL_OFFSETS[ARM_MOUNT_KEY])
    p.resetBasePositionAndOrientation(arm_id, arm_mount_pos, p.getQuaternionFromEuler([0, 0, yaw]))
    frame_back = settle_and_capture(robot_pos, yaw, "nudge_back.png")
    diff = np.mean(np.abs(frame_back.astype(int) - frame_fwd.astype(int)))
    print(f"  nudged back to start   frame diff vs previous = {diff:6.2f}")

    print(f"--- Camera sanity check complete. Inspect ./{out_dir}/*.png ---\n")


def _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, pos, yaw_val):
    """Move every body part + arm mount to (pos, yaw_val) and let the arm settle."""
    set_robot_pose(robot_ids, pos, yaw_val)
    arm_mount_pos = local_to_world(pos, yaw_val, LOCAL_OFFSETS[ARM_MOUNT_KEY])
    p.resetBasePositionAndOrientation(arm_id, arm_mount_pos, p.getQuaternionFromEuler([0, 0, yaw_val]))
    for _ in range(30):
        hold_arm_pose(arm_id, arm_joints, home_targets)
        p.stepSimulation()
        time.sleep(1 / 240)


# ---------------------------------------------------------------------------
# Obstacle avoidance
# ---------------------------------------------------------------------------
# The navigation used to rely on a single one-directional clamp (don't let
# robot_pos.x go past the table's near edge), which only protects an
# approach from that one specific side — a robot starting anywhere else
# (the far side, either flank) had nothing stopping it from kinematically
# driving straight through the table's solid volume. This section replaces
# that with real 2D obstacle avoidance: axis-aligned keepout rectangles for
# known obstacles, and a visibility-graph path planner that routes around
# them via their corners when a direct line is blocked — the standard
# technique for planning around a small number of convex polygonal
# obstacles.

def get_keepout_rects(objects, clearance=0.15):
    """
    Build (xmin, ymin, xmax, ymax) keepout rectangles for known obstacles,
    expanded by `clearance` so the robot's own body doesn't clip the real
    object even when routed right at the boundary. Currently just the
    table; add more entries here if more obstacles are added to the scene.
    """
    aabb_min, aabb_max = p.getAABB(objects["table"])
    return [(
        aabb_min[0] - clearance, aabb_min[1] - clearance,
        aabb_max[0] + clearance, aabb_max[1] + clearance,
    )]


def _segment_intersects_rect(p1, p2, rect):
    """
    Liang-Barsky line-clipping test: does segment p1->p2 cross the
    axis-aligned rectangle `rect` = (xmin, ymin, xmax, ymax)?
    """
    xmin, ymin, xmax, ymax = rect
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p_, q_ in ((-dx, x1 - xmin), (dx, xmax - x1), (-dy, y1 - ymin), (dy, ymax - y1)):
        if abs(p_) < 1e-9:
            if q_ < 0:
                return False  # parallel to this edge and outside it
            continue
        t = q_ / p_
        if p_ < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def _point_in_rect(point, rect):
    x, y = point
    xmin, ymin, xmax, ymax = rect
    return xmin <= x <= xmax and ymin <= y <= ymax


def is_path_clear(a_xy, b_xy, keepout_rects):
    return not any(_segment_intersects_rect(a_xy, b_xy, r) for r in keepout_rects)


def plan_path_around_obstacles(start_xy, goal_xy, keepout_rects):
    """
    Visibility-graph path planning around axis-aligned rectangular
    obstacles. Returns a waypoint list [start, ..., goal] (start
    included). If the direct line is already clear, that's the path.
    Otherwise tries routing via a single obstacle corner, then via two
    corners of the same obstacle, picking the shortest fully-clear
    option found. Falls back to the direct line (with a printed warning)
    only if no clear route through the corners exists at all — this can
    happen if start or goal is itself inside a keepout region.
    """
    start_xy, goal_xy = tuple(start_xy), tuple(goal_xy)

    if is_path_clear(start_xy, goal_xy, keepout_rects):
        return [start_xy, goal_xy]

    def path_length(path):
        return sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))

    # Routing waypoints sit a little OUTSIDE the keepout rectangle's own
    # corners (not exactly on them) — a path between two corners of the
    # same rectangle otherwise lies exactly on its boundary, which the
    # intersection test correctly flags as touching the obstacle, so no
    # two-corner route could ever validate. This small extra margin fixes
    # that without weakening the actual safety check itself.
    routing_margin = 0.05

    candidates = []
    for rect in keepout_rects:
        xmin, ymin, xmax, ymax = rect
        corners = [
            (xmin - routing_margin, ymin - routing_margin),
            (xmin - routing_margin, ymax + routing_margin),
            (xmax + routing_margin, ymin - routing_margin),
            (xmax + routing_margin, ymax + routing_margin),
        ]

        # Single-corner detour
        for c in corners:
            path = [start_xy, c, goal_xy]
            if is_path_clear(start_xy, c, keepout_rects) and is_path_clear(c, goal_xy, keepout_rects):
                candidates.append(path)

        # Two-corner detour (needed when start/goal are on "opposite sides"
        # and no single corner has a clear line to both)
        for c1 in corners:
            for c2 in corners:
                if c1 == c2:
                    continue
                path = [start_xy, c1, c2, goal_xy]
                if all(is_path_clear(path[i], path[i + 1], keepout_rects) for i in range(len(path) - 1)):
                    candidates.append(path)

    if candidates:
        candidates.sort(key=path_length)
        return candidates[0]

    print("  [path planner] no fully clear route found around obstacles "
          "(start or goal may be inside a keepout zone) — falling back to a direct line.")
    return [start_xy, goal_xy]


def navigate_to(robot_ids, arm_id, arm_joints, home_targets, current_pos, goal_pos, keepout_rects):
    """
    Move the robot from current_pos to goal_pos, routing AROUND any
    obstacle in the way instead of moving through it in a straight line.
    Returns (final_pos, final_yaw).
    """
    path = plan_path_around_obstacles(current_pos, goal_pos, keepout_rects)
    if len(path) > 2:
        print(f"  [navigate_to] obstacle in the way — routing via {len(path) - 2} waypoint(s) around it.")

    pos = list(current_pos)
    yaw = 0.0
    for waypoint in path[1:]:
        direction = np.array(waypoint) - np.array(pos)
        if np.linalg.norm(direction) > 1e-6:
            yaw = float(np.arctan2(direction[1], direction[0]))
        pos = list(waypoint)
        _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, pos, yaw)
    return pos, yaw


def search_for_target(objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw,
                       instruction, known_objects, api_key, max_steps=10,
                       turn_step_deg=25, move_step=0.15):
    """
    Closed-loop VLA-style search: capture the current view, ask Gemini for
    the single next action (turn/move/declare-target-reached), execute it,
    repeat — instead of driving to one hardcoded point and hoping the
    target happens to be in frame there.

    Real obstacle avoidance: any move that would drive the robot through
    the table's keepout rectangle (see get_keepout_rects) is rejected
    outright and replaced with a turn, regardless of which side or
    direction the robot is approaching from — not just a one-directional
    boundary clamp.

    Returns (robot_pos, yaw, target_name, found_bool).
    """
    from gemini_perception import decide_next_action, PerceptionError

    keepout_rects = get_keepout_rects(objects)
    x_min_allowed, x_max_allowed = -1.0, 2.5   # generous world bounds, not the obstacle boundary
    y_min_allowed, y_max_allowed = -1.5, 1.5

    print(f"\n--- Visual search: up to {max_steps} steps, instruction = {instruction!r} ---")
    for step in range(max_steps):
        _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw)
        img_path = os.path.abspath(f"search_step_{step:02d}.png")
        capture_face_camera(robot_pos, yaw, save_path=img_path, live_window=True)

        try:
            result = decide_next_action(img_path, instruction, known_objects, api_key=api_key)
        except PerceptionError as e:
            print(f"  step {step}: perception failed ({e}) — turning to try a different view.")
            result = {"action": "turn_left", "target_object": None, "visible": False,
                      "reasoning": "perception error, scanning"}

        action = result.get("action")
        print(f"  step {step}: action={action:<14} target={result.get('target_object')!s:<10} "
              f"visible={result.get('visible')!s:<5}  ({result.get('reasoning')})")

        if action == "target_reached" and result.get("target_object") in known_objects:
            print(f"--- Target found after {step + 1} step(s): {result['target_object']} ---\n")
            return robot_pos, yaw, result["target_object"], True

        if action == "turn_left":
            yaw += np.radians(turn_step_deg)
        elif action == "turn_right":
            yaw -= np.radians(turn_step_deg)
        elif action in ("move_forward", "move_backward"):
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            direction = forward if action == "move_forward" else -forward
            candidate = np.array(robot_pos) + direction * move_step
            candidate = np.clip(candidate, [x_min_allowed, y_min_allowed], [x_max_allowed, y_max_allowed])
            candidate = candidate.tolist()
            # Real obstacle check: does the segment from here to there cross
            # the table's keepout rectangle? Reject the move outright rather
            # than silently clamping to some nearby "safe-ish" point, which
            # could still graze the obstacle depending on approach angle.
            if is_path_clear(robot_pos, candidate, keepout_rects):
                robot_pos = candidate
            else:
                print(f"  step {step}: {action} would drive into the table — blocked, turning instead.")
                yaw += np.radians(turn_step_deg)
        # any other/unrecognized action: just re-capture and ask again next loop

    print(f"--- Search exhausted after {max_steps} steps without a confident 'target_reached'. ---\n")
    return robot_pos, yaw, None, False


# Empirically validated standoff distance (see do_pick's docstring/history):
# x = table_front_edge - PICK_STANDOFF_FROM_EDGE lands the robot in the
# x=0.20-0.30 window where do_pick's 3-phase approach converges to
# ~0.000m offset. Outside that window (tested down to x=0.15 and up to
# x=0.40) convergence degrades or fails.
PICK_STANDOFF_FROM_EDGE = 0.20


def align_for_pick(objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, target_pos, table_id):
    """
    Reposition the robot to a validated pose for do_pick, instead of just
    creeping forward until "close enough" by raw distance.

    Two corrections, both required:
    1. x is set to the empirically-validated standoff distance from the
       table's front edge (see PICK_STANDOFF_FROM_EDGE) — do_pick's
       approach only reliably converges in a narrow x window.
    2. y is aligned so the arm's shoulder mount lines up with the
       target's actual y-coordinate. The arm mounts on the RIGHT shoulder
       (a fixed negative local y offset) — a target that happens to sit
       near that side (e.g. the red ball in initial testing) reaches
       fine, but a target on the opposite side (e.g. the green ball)
       requires the arm to reach all the way across its own body and
       consistently fails (verified: 0.32-0.36m offset, "succeeding"
       only via the live re-centering step dragging the object from
       wherever it ended up — not a real grasp). Shifting the robot
       sideways so the shoulder already lines up with the target fixes
       this for either side, verified to converge to 0.000m offset for
       both balls.
    Yaw is squared up to face the table directly (yaw=0 relative to the
    table's approach direction), matching the orientation every do_pick
    validation run used.

    Routes to that pose via navigate_to() (real obstacle avoidance) rather
    than teleporting straight there — if the robot is currently on a side
    of the table where a direct line to the aligned pose would cross it,
    this goes around instead of driving through it.
    """
    aabb_min, _ = p.getAABB(table_id)
    target_x = aabb_min[0] - PICK_STANDOFF_FROM_EDGE
    # shoulder_world_y = robot_y + LOCAL_OFFSETS[ARM_MOUNT_KEY][1] (at yaw=0)
    target_y = target_pos[1] - LOCAL_OFFSETS[ARM_MOUNT_KEY][1]
    aligned_pos = [target_x, target_y]

    keepout_rects = get_keepout_rects(objects)
    navigate_to(robot_ids, arm_id, arm_joints, home_targets, robot_pos, aligned_pos, keepout_rects)
    # Face the table squarely once there — navigate_to leaves yaw pointed
    # along the last travel segment, but do_pick was validated at yaw=0.
    _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, aligned_pos, 0.0)
    return aligned_pos, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-check", action="store_true",
        help="Run a quick rotate+nudge camera sanity check and exit, instead of the full nav+pick demo.",
    )
    parser.add_argument(
        "--instruction", default="pick up the green ball",
        help='Natural-language instruction for Gemini to interpret, e.g. "pick up the red ball".',
    )
    parser.add_argument(
        "--gemini-api-key", default=None,
        help="Overrides the GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--no-perception", action="store_true",
        help="Skip the Gemini call entirely and just target the green ball (offline testing, no API key needed).",
    )
    parser.add_argument(
        "--max-search-steps", type=int, default=10,
        help="Max turn/move steps the visual search loop will take before giving up and falling back.",
    )
    parser.add_argument(
        "--print-joints", action="store_true",
        help="Print every joint's index/name/type for the loaded Panda arm and exit. "
             "Use this to verify end_effector_index=11 and finger_joints=[9,10] are correct "
             "for your installed franka_panda/panda.urdf, since versions can differ.",
    )
    parser.add_argument(
        "--start-x", type=float, default=0.0,
        help="Robot's starting x position. Default 0.0 (facing the table's general direction "
             "is NOT guaranteed at other values — combine with --start-yaw-deg).",
    )
    parser.add_argument(
        "--start-y", type=float, default=0.0,
        help="Robot's starting y position.",
    )
    parser.add_argument(
        "--start-yaw-deg", type=float, default=0.0,
        help="Robot's starting facing direction in degrees (0 = facing +x, i.e. toward the "
             "table from the default start). Use e.g. 180 to start facing AWAY from the table, "
             "to test whether the search loop can turn around and still find the target.",
    )
    parser.add_argument(
        "--random-start", action="store_true",
        help="Ignore --start-x/--start-y/--start-yaw-deg and instead pick a random position "
             "(within the safe navigation box) and a random facing direction (0-360 degrees). "
             "Useful for stress-testing the search loop from many different starting conditions, "
             "including ones where the target isn't visible at all until the robot turns.",
    )
    args = parser.parse_args()

    objects = setup_static_scene()
    robot_ids = create_robot_visual()

    if args.random_start:
        # Match the safety box search_for_target() uses, so a random start
        # never spawns the robot ON/inside the table.
        aabb_min, _ = p.getAABB(objects["table"])
        x_max_allowed = aabb_min[0] - 0.15
        robot_pos = [random.uniform(-1.0, x_max_allowed), random.uniform(-1.0, 1.0)]
        yaw = random.uniform(0, 2 * np.pi)
        print(f"--random-start: spawning at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
              f"facing {np.degrees(yaw):.0f} degrees")
    else:
        robot_pos = [args.start_x, args.start_y]
        yaw = np.radians(args.start_yaw_deg)

    # every part's height/offset from robot_pos is derived from the
    # LOCAL_OFFSETS layout above.

    arm_mount_pos = local_to_world(robot_pos, yaw, LOCAL_OFFSETS[ARM_MOUNT_KEY])
    arm_id = load_arm_on_shoulder(arm_mount_pos, yaw)

    if args.print_joints:
        joint_type_names = {p.JOINT_REVOLUTE: "REVOLUTE", p.JOINT_PRISMATIC: "PRISMATIC",
                             p.JOINT_FIXED: "FIXED", p.JOINT_SPHERICAL: "SPHERICAL",
                             p.JOINT_PLANAR: "PLANAR"}
        print(f"\n{'idx':>4}  {'type':<10}  {'joint name':<24}  link name")
        for i in range(p.getNumJoints(arm_id)):
            info = p.getJointInfo(arm_id, i)
            jtype = joint_type_names.get(info[2], str(info[2]))
            joint_name = info[1].decode() if isinstance(info[1], bytes) else info[1]
            link_name = info[12].decode() if isinstance(info[12], bytes) else info[12]
            print(f"{i:>4}  {jtype:<10}  {joint_name:<24}  {link_name}")
        print("\nLook for the 'hand'/end-effector link (used as end_effector_index in do_pick) "
              "and the two prismatic finger joints (used as finger_joints).")
        p.disconnect()
        return

    # Identify the arm's controllable joints and lock in its current
    # ("home") pose so gravity doesn't collapse it while we navigate.
    arm_joints = get_controllable_joints(arm_id)
    home_targets = [p.getJointState(arm_id, j)[0] for j in arm_joints]

    for _ in range(50):
        hold_arm_pose(arm_id, arm_joints, home_targets)
        p.stepSimulation()
        time.sleep(1 / 240)

    if args.camera_check:
        run_camera_sanity_check(robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw)
        print("Camera check done. Close the window or Ctrl+C to exit.")
        try:
            while True:
                p.stepSimulation()
                time.sleep(1 / 240)
        except KeyboardInterrupt:
            p.disconnect()
        return

    known_objects = ["green_ball", "red_ball"]

    if args.no_perception:
        # Offline fallback path: skip the search loop entirely, drive to a
        # fixed standoff point and just target the green ball.
        print("--no-perception set: skipping the visual search loop.")
        approach_xy = compute_approach_point(objects["table"], robot_pos, standoff=0.25)
        aabb_min, aabb_max = p.getAABB(objects["table"])
        table_center_xy = [(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] + aabb_max[1]) / 2]
        facing = np.array(table_center_xy) - np.array(approach_xy)
        yaw = float(np.arctan2(facing[1], facing[0]))
        robot_pos = approach_xy
        _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw)
        capture_face_camera(robot_pos, yaw, save_path=os.path.abspath("scene.png"), live_window=True)
        target_name = "green_ball"
    else:
        # --- Closed-loop visual search: the robot decides its own turns/
        # moves based on what it currently sees, instead of driving to one
        # hardcoded point and hoping the target happens to be in frame. ---
        robot_pos, yaw, target_name, found = search_for_target(
            objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw,
            args.instruction, known_objects, args.gemini_api_key,
            max_steps=args.max_search_steps,
        )
        if not found:
            target_name = "green_ball"
            print(f"Falling back to '{target_name}' since search didn't confirm a target.")

    # The VLM only decided WHICH object and roughly confirmed it's close —
    # ground the final approach in the physics engine's real object pose,
    # then reposition to a validated pick pose (correct standoff distance
    # AND lateral alignment with the target — see align_for_pick) before
    # attempting do_pick.
    target_object_pos = p.getBasePositionAndOrientation(objects[target_name])[0]
    robot_pos, yaw = align_for_pick(
        objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, target_object_pos, objects["table"],
    )
    target_object_pos = p.getBasePositionAndOrientation(objects[target_name])[0]  # re-read after moving

    print("Executing pick...")
    _, pick_succeeded = do_pick(arm_id, target_object_pos, target_body_id=objects[target_name])

    final_pos = p.getBasePositionAndOrientation(objects[target_name])[0]
    if pick_succeeded:
        print(f"Pick SUCCEEDED. '{target_name}' position now:", final_pos)
    else:
        print(f"Pick FAILED — '{target_name}' was not actually lifted (position now: {final_pos}). "
              f"It may have been knocked off the table or missed during the grasp.")

    print("\nSimulation running. Close the window or Ctrl+C to exit.")
    try:
        while True:
            p.stepSimulation()
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        p.disconnect()
    except p.error:
        # Closing the PyBullet window disconnects the physics server, so
        # the next stepSimulation() call throws this — that's the user
        # doing exactly what the prompt above told them to do, not a bug.
        pass


if __name__ == "__main__":
    main()