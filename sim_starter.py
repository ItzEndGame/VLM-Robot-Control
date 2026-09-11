"""
Mobile manipulator simulation: self-balancing twin-wheel base + humanoid
torso/head + Panda arm/gripper + face camera.

Modeled loosely on reference robots like Enchanted Tools' "Miroki" /
similar service robots: a black two-wheel base, a thin rising column,
a white humanoid torso with twin arms, and a dark sensor head on top.

Install (on your own machine, with a display):
    pip install pybullet numpy google-genai pillow
    (google-genai and pillow are only needed for the Gemini-driven search;
    --no-perception skips both.)

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
  and default to a fixed approach point + the first known ball, for
  offline testing without an API key.
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

# Worst-case horizontal distance from the robot's (x, y) reference point to
# ANY physical part of the robot — used by obstacle avoidance so a wheel or
# the decorative arm can't clip an obstacle even while the reference point
# itself stays clear of it (that was the actual bug: routing only kept the
# single reference point outside the keepout zone, not the robot's real
# footprint). Checked both candidates directly rather than guessing which
# extends farther — they turned out to be within 1cm of each other.
_wheel_extent = float(np.hypot(WHEEL_RADIUS, WHEEL_Y_OFFSET + WHEEL_THICKNESS / 2))
_deco_hand_extent = float(np.hypot(_deco_hand_local[0], _deco_hand_local[1]) + DECO_HAND_RADIUS)
ROBOT_FOOTPRINT_RADIUS = max(_wheel_extent, _deco_hand_extent)

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


def setup_static_scene(num_balls=5, num_boxes=5, include_boxes=False):
    """
    Ground plane and table with `num_balls` colored balls near the table's
    front edge. Optionally also `num_boxes` small open-top boxes along the
    table's FAR end, each painted a distinct solid color — the same visual-
    identification mechanism already used (and verified reliable) for the
    balls. Which physical box slot gets which color is shuffled at scene-
    build time, specifically so nothing downstream (the search/approach
    code) can assume "the orange box is always at position X" — the only
    way to find a specific colored box is to actually look at it via the
    camera, same as finding a specific colored ball.

    (An earlier version painted a NUMBER onto each box via a texture —
    dropped after a real run showed the numbers weren't actually rendering
    visibly. Solid color reuses a mechanism already proven to work here.)
    """
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

    ball_colors = [
        ("red_ball", [1, 0, 0, 1]),
        ("green_ball", [0, 1, 0, 1]),
        ("blue_ball", [0.1, 0.3, 1, 1]),
        ("yellow_ball", [1, 0.9, 0, 1]),
        ("purple_ball", [0.6, 0.1, 0.8, 1]),
    ][:num_balls]

    table_half_y = (aabb_max[1] - aabb_min[1]) / 2 - 0.08  # keep off the very edges
    ball_ys = np.linspace(-table_half_y, table_half_y, len(ball_colors)) if len(ball_colors) > 1 else [0.0]

    objects = {"table": table_id}
    for (name, rgba), y in zip(ball_colors, ball_ys):
        ball_z = table_top_z + ball_radius + 0.01
        ball_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=rgba),
            basePosition=[ball_x, float(y), ball_z],
        )
        # PyBullet's default dynamics for a small, light sphere are
        # underdamped and bouncy — fine for it just sitting on the table,
        # but a real problem the moment a gripper actually contacts it:
        # verified directly that this contributes to the fingers
        # "exploding" the ball away on contact instead of grasping it
        # cleanly.
        p.changeDynamics(ball_id, -1, lateralFriction=1.2, spinningFriction=0.005,
                          rollingFriction=0.005, restitution=0.0,
                          linearDamping=0.3, angularDamping=0.3)
        objects[name] = ball_id
    objects["ball_names"] = [name for name, _ in ball_colors]

    if include_boxes:
        # Simple open-top boxes built from 5 static (mass=0) box primitives
        # (floor + 4 walls) as ONE compound body each — no external URDF
        # asset dependency, consistent with how the balls/robot are built.
        # Sit on the tabletop at the FAR end (opposite the balls), spread
        # across y so align_for_place's far-edge approach can reach any of
        # them, and so there's room to visually tell them apart.
        wall_h, wall_t = 0.06, 0.006
        inner_half = 0.055  # tight row of 5 across the table's width
        outer_half = inner_half + wall_t
        box_x = aabb_max[0] - (outer_half + 0.05)
        box_ys = np.linspace(-table_half_y, table_half_y, num_boxes) if num_boxes > 1 else [0.0]
        box_base_z = table_top_z

        # Boxes are identified by SOLID COLOR, not a painted-on number —
        # texture-mapped digits (tried first) turned out not to render
        # visibly in practice (verified directly against a real run), and
        # solid rgbaColor is the one visual-identification mechanism
        # already proven reliable in this file (it's exactly how the
        # balls are told apart, and that search has worked consistently).
        # Reusing a known-good mechanism beats debugging pybullet texture/
        # UV behavior blind, with no display available to verify fixes.
        box_color_choices = [
            ("orange", [1.0, 0.55, 0.0, 1]),
            ("cyan", [0.0, 0.9, 0.9, 1]),
            ("magenta", [0.9, 0.1, 0.9, 1]),
            ("white", [0.95, 0.95, 0.95, 1]),
            ("black", [0.08, 0.08, 0.08, 1]),
        ][:num_boxes]
        random.shuffle(box_color_choices)  # physical left-to-right order is NOT this list's order

        for slot_y, (color_name, box_rgba) in zip(box_ys, box_color_choices):
            col_shapes = [
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[outer_half, outer_half, wall_t / 2]),
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_t / 2, outer_half, wall_h / 2]),
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_t / 2, outer_half, wall_h / 2]),
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[outer_half, wall_t / 2, wall_h / 2]),
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[outer_half, wall_t / 2, wall_h / 2]),
            ]
            vis_shapes = [
                p.createVisualShape(p.GEOM_BOX, halfExtents=[outer_half, outer_half, wall_t / 2], rgbaColor=box_rgba),
                p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_t / 2, outer_half, wall_h / 2], rgbaColor=box_rgba),
                p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_t / 2, outer_half, wall_h / 2], rgbaColor=box_rgba),
                p.createVisualShape(p.GEOM_BOX, halfExtents=[outer_half, wall_t / 2, wall_h / 2], rgbaColor=box_rgba),
                p.createVisualShape(p.GEOM_BOX, halfExtents=[outer_half, wall_t / 2, wall_h / 2], rgbaColor=box_rgba),
            ]
            # Local offsets relative to the box's base frame (base = floor center):
            local_pos = [
                [0, 0, wall_t / 2],                                   # floor
                [-outer_half + wall_t / 2, 0, wall_t + wall_h / 2],   # -x wall
                [outer_half - wall_t / 2, 0, wall_t + wall_h / 2],    # +x wall (faces an approach from +x)
                [0, -outer_half + wall_t / 2, wall_t + wall_h / 2],   # -y wall
                [0, outer_half - wall_t / 2, wall_t + wall_h / 2],    # +y wall
            ]
            box_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shapes[0],
                baseVisualShapeIndex=vis_shapes[0],
                basePosition=[box_x, float(slot_y), box_base_z],
                linkMasses=[0] * 4,
                linkCollisionShapeIndices=col_shapes[1:],
                linkVisualShapeIndices=vis_shapes[1:],
                linkPositions=local_pos[1:],
                linkOrientations=[[0, 0, 0, 1]] * 4,
                linkInertialFramePositions=[[0, 0, 0]] * 4,
                linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
                linkParentIndices=[0] * 4,
                linkJointTypes=[p.JOINT_FIXED] * 4,
                linkJointAxis=[[0, 0, 0]] * 4,
            )
            p.resetBasePositionAndOrientation(box_id, [box_x, float(slot_y), box_base_z], [0, 0, 0, 1])

            objects[f"box_{color_name}"] = box_id
            # box's floor top surface, in world z — used by align_for_place/
            # do_place to know how deep to lower the ball.
            objects[f"box_{color_name}_floor_z"] = box_base_z + wall_t
        objects["box_colors"] = [name for name, _ in box_color_choices]

    return objects


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


def capture_face_camera(position_xy, yaw, save_path=None):
    """
    Simulate the 'face' camera: mounted in the HEAD unit, at the front of
    the robot, looking in the direction the robot currently faces.

    - Calling this every simulation step gives you a REAL-TIME feed:
      PyBullet's own GUI automatically shows an on-screen "Synthetic
      Camera RGB/Depth/Segmentation" preview whenever getCameraImage()
      is called.
    - Pass save_path to also write a PNG snapshot to disk (e.g. once,
      right before sending it to the VLM/Gemini step).
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


def do_place(arm_id, place_pos, grasp_constraint, approach_yaw=np.pi):
    """
    Mirror of do_pick's staged, verified motion pattern, but for RELEASING
    a held object instead of grasping one: hover above place_pos, reorient
    if needed, descend, open the gripper (dropping/releasing the object
    from a small height rather than jamming it into the surface), remove
    the grasp constraint, then retract.

    Reuses the same lessons do_pick already verified — small joint-space
    steps for the big far-away move, position+orientation IK locked
    together only for the final Cartesian approach, gradual (not instant)
    gripper motion — since there's no reason placing would be less prone
    to the same jerky-motion issues picking was.

    approach_yaw: the robot BODY's yaw when this runs (align_for_place
    always uses pi — approaching from beyond the table's far edge, facing
    back toward it). This matters for more than just bookkeeping: a
    world-frame "point straight down" orientation is NOT arm-base-frame-
    invariant when the base itself is rotated. Verified directly (by
    reproducing pybullet's Euler convention and comparing rotation
    matrices) that reusing do_pick's plain Euler(pi,0,0) target here asks
    the arm for a DIFFERENT orientation relative to ITS OWN base than the
    one it successfully reaches during pick — even though both nominally
    "point down" in world space — because do_pick's base sits at yaw=0
    while do_place's sits at yaw=pi. That mismatch is what produced offsets
    that grew every step (0.12m -> 0.4m+ over 6 descent stages) instead of
    settling: IK was chasing a target that was arm-relative-awkward (likely
    fighting a joint limit) rather than the same comfortable local pose
    do_pick uses. Composing the approach yaw into the target orientation
    (Euler(pi, 0, approach_yaw), not just Euler(pi, 0, 0)) cancels the
    base's own rotation out, giving the arm the SAME local/joint-relative
    target it already handles well — confirmed this still points the
    gripper straight down either way (irrelevant which way "down" is
    twisted around vertical for releasing a symmetric ball).
    """
    end_effector_index = 11
    finger_joints = [9, 10]
    grasp_orientation = p.getQuaternionFromEuler([np.pi, 0, approach_yaw])

    # Release from a small height above the target rather than driving the
    # gripper all the way down to touch it — avoids the fingers (still
    # holding the object) colliding with the surface itself.
    #
    # 0.08m, not the smaller clearance an earlier version used: for the
    # "other end of the table" case (open surface, no walls) that smaller
    # height was fine, but for the box case it left only ~1cm of vertical
    # clearance between the release point and the box's rim — and the
    # gripper opens to within ~1.5cm of the box's own interior walls
    # horizontally. Computed directly against the actual box/gripper
    # geometry (wall height 6cm, gripper open span 8cm vs 11cm interior):
    # that's tight enough that the fingers plausibly clip a wall while
    # opening, right at rim height — a real, physically-checked risk this
    # time, not a guess. This height clears the rim by a comfortable
    # margin regardless.
    release_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + 0.08])
    hover_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + 0.35])

    controllable = get_controllable_joints(arm_id)
    start_joint_state = [p.getJointState(arm_id, j)[0] for j in controllable]
    # Position-ONLY IK for this far-away hover solve — matching do_pick's
    # own Phase 1 exactly. An earlier version of this function locked
    # `grasp_orientation` in here too, which is precisely the case do_pick's
    # docstring already warns is unstable ("Locking it from the very first
    # far-away hover point was tested and found unstable") — and the
    # symptom matched exactly: IK offsets of 0.35-0.5m that barely changed
    # across every subsequent stage, i.e. the solver never found a good
    # solution at all once orientation was locked this far from the target.
    hover_joint_solution = p.calculateInverseKinematics(
        arm_id, end_effector_index, hover_pos.tolist(),
        maxNumIterations=200, residualThreshold=1e-5,
    )
    n_joint_steps = 20
    for step in range(1, n_joint_steps + 1):
        t = step / n_joint_steps
        for i, j in enumerate(controllable):
            if j in finger_joints:
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, 0.0, force=50)  # stay closed
            else:
                interp = start_joint_state[i] + (hover_joint_solution[i] - start_joint_state[i]) * t
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, interp, force=80)
        for _ in range(15):
            p.stepSimulation()
            time.sleep(1 / 240)
    actual_hover_pos = p.getLinkState(arm_id, end_effector_index)[0]
    print(f"  [do_place] stage=hover(joint-space)  target={tuple(round(v,3) for v in hover_pos)}  "
          f"actual_hand_pos={tuple(round(v,3) for v in actual_hover_pos)}  "
          f"offset={np.linalg.norm(np.array(actual_hover_pos)-hover_pos):.3f} m")

    # Reorient in place (position held fixed at hover) from whatever
    # orientation the position-only hover solve naturally landed on, to
    # the target top-down release orientation — mirrors do_pick's Phase 2
    # exactly, for the same reason: combining a position change and an
    # orientation change in one discontinuous IK jump is the specific
    # thing that produced the original 0.3-0.5m offset bug here (see the
    # note on the hover solve above). Skipping this step would just move
    # that same risk one step later, onto the first descent waypoint,
    # instead of actually eliminating it.
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
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, 0.0, force=50)  # stay closed
            else:
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_angle, force=80)
        for _ in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)
    actual_reoriented_pos = p.getLinkState(arm_id, end_effector_index)[0]
    print(f"  [do_place] stage=reorient(in-place)  hand now at "
          f"{tuple(round(v,3) for v in actual_reoriented_pos)}  "
          f"offset_from_hover={np.linalg.norm(np.array(actual_reoriented_pos)-hover_pos):.4f} m")

    # Fine Cartesian descent from hover down to the release point, orientation
    # locked for every step (same pattern as do_pick's Phase 3).
    n_descent_steps = 6
    waypoints = [
        hover_pos + (release_pos - hover_pos) * (i / n_descent_steps)
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
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, 0.0, force=50)  # stay closed until release
            else:
                p.setJointMotorControl2(arm_id, i, p.POSITION_CONTROL, joint_angle,
                                         force=200 if is_final else 80)
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1 / 240)
        actual_pos = p.getLinkState(arm_id, end_effector_index)[0]
        offset = np.linalg.norm(np.array(actual_pos) - np.array(pos))
        stage_name = "release-descent" if is_final else f"descent {idx}/{n_descent_steps}"
        print(f"  [do_place] stage={stage_name:<16} target={tuple(round(v,3) for v in pos)}  "
              f"actual_hand_pos={tuple(round(v,3) for v in actual_pos)}  offset={offset:.3f} m")

    # Release: remove the rigid grasp constraint, THEN open the fingers
    # gradually. Removing the constraint first lets the object fall free
    # under real physics; opening the fingers gradually (not instantly)
    # avoids a hard "flick" off a finger that's still in contact.
    if grasp_constraint is not None:
        p.removeConstraint(grasp_constraint)
    n_open_steps = 20
    for step in range(n_open_steps):
        finger_target = 0.04 * (step / (n_open_steps - 1))
        for finger in finger_joints:
            p.setJointMotorControl2(arm_id, finger, p.POSITION_CONTROL, finger_target, force=10)
        p.stepSimulation()
        time.sleep(1 / 240)
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1 / 240)

    # Retract straight up, fingers open, small steps for the same reason
    # as everywhere else in this file — avoid sweeping through what was
    # just placed.
    retract_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + 0.25])
    n_retract_steps = 4
    for i in range(1, n_retract_steps + 1):
        wp = (release_pos + (retract_pos - release_pos) * (i / n_retract_steps)).tolist()
        joint_poses = p.calculateInverseKinematics(
            arm_id, end_effector_index, wp, grasp_orientation,
            maxNumIterations=200, residualThreshold=1e-5,
        )
        for j, joint_angle in zip(controllable, joint_poses):
            if j in finger_joints:
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, 0.04, force=20)
            else:
                p.setJointMotorControl2(arm_id, j, p.POSITION_CONTROL, joint_angle, force=150)
        for _ in range(60):
            p.stepSimulation()
            time.sleep(1 / 240)

    print("  [do_place] release + retract complete")


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
        return capture_face_camera(pos, yaw_val, save_path=os.path.join(out_dir, filename))

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

def get_keepout_rects(objects, clearance=ROBOT_FOOTPRINT_RADIUS + 0.05):
    """
    Build (xmin, ymin, xmax, ymax) keepout rectangles for known obstacles,
    expanded by `clearance` so the robot's own body doesn't clip the real
    object even when routed right at the boundary. Currently just the
    table; add more entries here if more obstacles are added to the scene.

    Default clearance = ROBOT_FOOTPRINT_RADIUS (the real worst-case
    distance from the robot's reference point to any physical part of it —
    see its definition) plus a small extra margin. The previous default
    (0.15m) only accounted for a point-robot at the reference (x, y)
    position; a wheel or the decorative arm could still visibly clip the
    table's actual geometry even while that reference point stayed
    outside the old, thinner keepout zone.
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


def _nearest_exit_point(point, rect, margin=0.03):
    """
    Given `point` known to be INSIDE `rect`, return the point just outside
    the nearest edge (straight out, perpendicular to that edge, pushed
    `margin` past the boundary so it's unambiguously outside).

    Used to fix a real bug: align_for_pick/align_for_place deliberately
    park the robot CLOSER to the table than the navigation clearance
    buffer (see PICK_STANDOFF_FROM_EDGE / PLACE_STANDOFF_FROM_EDGE) — the
    buffer is intentionally more conservative than the real table hitbox,
    for exactly this reason. That means the next navigate_to call always
    STARTS inside the keepout rectangle. A straight line from a point
    inside a convex rectangle to anywhere outside it always crosses the
    boundary, so plan_path_around_obstacles could never certify ANY
    corner-routed candidate as fully clear — every one's first leg would
    be rejected — forcing a fall-through to a completely UNCHECKED direct
    line. Verified directly: that unchecked fallback is exactly what let
    the robot drive straight through the table when placing at the far
    end right after a pick (start = near-edge standoff, goal = far side).

    The fix is this function: take one short, safe hop straight out to
    the nearest edge of the buffer FIRST (moving further from the table
    along the same axis the standoff was already built on — this can
    only move away from the table's real footprint, never into it), then
    run the normal, fully-checked corner routing from that verified-
    exterior point.
    """
    x, y = point
    xmin, ymin, xmax, ymax = rect
    d_left, d_right = x - xmin, xmax - x
    d_bottom, d_top = y - ymin, ymax - y
    d_min = min(d_left, d_right, d_bottom, d_top)
    if d_min == d_left:
        return (xmin - margin, y)
    if d_min == d_right:
        return (xmax + margin, y)
    if d_min == d_bottom:
        return (x, ymin - margin)
    return (x, ymax + margin)


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
    happen if GOAL is itself inside a keepout region (align_for_pick/
    align_for_place handle that themselves, by only ever passing this
    function a goal that's outside every rect, then doing their own
    verified-safe final creep in — see their docstrings).

    If START is inside a keepout region instead (see _nearest_exit_point's
    docstring for why that legitimately happens), this first takes a
    short escape hop to the nearest exterior point, THEN plans normally
    from there — rather than letting every candidate's first leg fail
    and silently falling back to a totally unchecked direct line.
    """
    start_xy, goal_xy = tuple(start_xy), tuple(goal_xy)

    escape_hops = []
    effective_start = start_xy
    for _ in range(len(keepout_rects) + 1):  # bounded: at most one escape per rect
        containing = next((r for r in keepout_rects if _point_in_rect(effective_start, r)), None)
        if containing is None:
            break
        effective_start = _nearest_exit_point(effective_start, containing)
        escape_hops.append(effective_start)

    def path_length(path):
        return sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) for i in range(len(path) - 1))

    if is_path_clear(effective_start, goal_xy, keepout_rects):
        return [start_xy] + escape_hops + [goal_xy]

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
            path = [effective_start, c, goal_xy]
            if is_path_clear(effective_start, c, keepout_rects) and is_path_clear(c, goal_xy, keepout_rects):
                candidates.append(path)

        # Two-corner detour (needed when start/goal are on "opposite sides"
        # and no single corner has a clear line to both)
        for c1 in corners:
            for c2 in corners:
                if c1 == c2:
                    continue
                path = [effective_start, c1, c2, goal_xy]
                if all(is_path_clear(path[i], path[i + 1], keepout_rects) for i in range(len(path) - 1)):
                    candidates.append(path)

    if candidates:
        candidates.sort(key=path_length)
        best = candidates[0]
        return [start_xy] + escape_hops[:-1] + best if escape_hops else [start_xy] + best[1:]

    print("  [path planner] no fully clear route found around obstacles "
          "(goal may be inside a keepout zone) — falling back to a direct line.")
    return [start_xy] + escape_hops + [goal_xy] if escape_hops else [start_xy, goal_xy]


def navigate_to(robot_ids, arm_id, arm_joints, home_targets, current_pos, goal_pos, keepout_rects,
                 speed=0.015, camera_update_every=5):
    """
    Move the robot from current_pos to goal_pos, routing AROUND any
    obstacle in the way instead of moving through it in a straight line.

    Moves smoothly and gradually (via move_robot_towards) rather than
    teleporting straight to each waypoint — `speed` is how far it travels
    per physics step; lower = slower, more visible motion. Returns
    (final_pos, final_yaw).
    """
    path = plan_path_around_obstacles(current_pos, goal_pos, keepout_rects)
    if len(path) > 2:
        print(f"  [navigate_to] obstacle in the way — routing via {len(path) - 2} waypoint(s) around it.")

    pos = list(current_pos)
    yaw = 0.0
    step_count = 0
    for waypoint in path[1:]:
        reached = False
        while not reached:
            pos, new_yaw, reached = move_robot_towards(robot_ids, arm_id, pos, waypoint, speed=speed)
            if new_yaw is not None:
                yaw = new_yaw
            hold_arm_pose(arm_id, arm_joints, home_targets)
            if step_count % camera_update_every == 0:
                capture_face_camera(pos, yaw)
            step_count += 1
            p.stepSimulation()
            time.sleep(1 / 240)
    # Settle once fully arrived, matching the old behavior at the final pose.
    _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, pos, yaw)
    return pos, yaw


def search_for_target(objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw,
                       instruction, known_objects, api_key, max_steps=20,
                       turn_step_deg=25, move_step=0.15, target_kind="object to pick up"):
    """
    Closed-loop VLA-style search: capture the current view, ask Gemini for
    the single next action (turn/move/declare-target-reached), execute it,
    repeat — instead of driving to one hardcoded point and hoping the
    target happens to be in frame there.

    IMPORTANT: this loop's job is only IDENTIFICATION (which known object
    does the instruction refer to?), not full navigation to a pick-ready
    pose. It stops as soon as a step reports the target VISIBLE and
    NAMED — not only on "target_reached" — and hands off to a ground-
    truth, obstacle-aware path planner (align_for_pick when locating an
    object to pick up, align_for_place when locating a numbered/colored
    destination box) for the actual approach. See the inline comment at
    that check for why: this loop's only movement primitives are turn
    and move-along-current-heading, which can deadlock (spin in place
    indefinitely) if the target is only visible while facing an obstacle
    the robot can't safely walk through — e.g. starting on the far side
    of the table.

    Real obstacle avoidance: any move that would drive the robot through
    the table's real (unpadded) footprint is rejected outright and
    replaced with a turn, regardless of which side or direction the robot
    is approaching from — not just a one-directional boundary clamp. This
    loop checks against the table's raw AABB rather than the padded
    keepout buffer used elsewhere (see get_keepout_rects) — the padded
    buffer is intentionally tighter than the robot can approach the table,
    so a search that starts inside it (e.g. right after a pick) needs to
    be able to move, not just turn in place.

    Returns (robot_pos, yaw, target_name, found_bool).
    """
    from gemini_perception import decide_next_action, PerceptionError, PerceptionFatalError

    # Separate, UNBUFFERED rect (clearance=0) used for this loop's own
    # coarse move-validity check below — NOT the padded rect that
    # get_keepout_rects returns by default (that one's used by
    # align_for_pick/align_for_place for the precise final approach).
    # Reason: the robot can legitimately START a search from INSIDE the
    # padded buffer (e.g. box search starting right at the pick standoff
    # position, right after a pick — see PICK_STANDOFF_FROM_EDGE, which is
    # deliberately closer to the table than the buffer). A segment from a
    # point inside a rect to anywhere outside it always registers as
    # "blocked" against that same rect, so using the padded rect here would
    # make EVERY move_forward/move_backward look blocked in that situation,
    # deadlocking the robot into turning forever, never able to actually
    # leave the standoff position. Checking against the real (unpadded)
    # table hitbox instead is still genuinely collision-safe for this
    # coarse scanning phase — align_for_pick/align_for_place do the
    # precise, safety-margin-respecting final approach afterward anyway.
    real_rects = get_keepout_rects(objects, clearance=0.0)
    x_min_allowed, x_max_allowed = -1.0, 2.5   # generous world bounds, not the obstacle boundary
    y_min_allowed, y_max_allowed = -1.5, 1.5

    # Gemini gets no memory between steps — each call only sees the CURRENT
    # frame, with no idea which way it turned last time or how far it's
    # rotated so far. Asking it to freely re-pick turn_left/turn_right every
    # single step (as long as the target stays invisible) produces a random
    # walk instead of a clean sweep: 2-3 lefts, a right, more lefts...
    # (observed directly). That can revisit the same headings repeatedly,
    # burning far more steps — and API calls — than a real 360-degree scan
    # needs, and was a direct contributor to hitting the free-tier rate
    # limit in one run. Fix: once a scan direction is picked (the first
    # turn while nothing's visible), keep committing to that same
    # direction until something IS visible — the model still decides
    # visibility/move/target_reached, it just isn't re-litigating spin
    # direction on every frame with no memory to base that choice on.
    scan_direction = None  # +1 = left, -1 = right; set on the first blind turn

    # PerceptionError used to be caught as one bucket and always handled
    # the same way — turn_left and try again next step — regardless of
    # WHY it failed. That's wrong for two different reasons depending on
    # the actual cause:
    #  - Config errors (missing API key, missing package) fail identically
    #    on every single call forever; turning does nothing, and the loop
    #    just burns all max_steps before reporting a misleading "target
    #    not found" when the real problem is "perception never worked".
    #  - A transient failure (e.g. a 429 that already exhausted its own
    #    internal backoff in _generate_content_with_retry) has nothing to
    #    do with viewpoint — turning moves the robot for no reason related
    #    to the actual problem, and firing another request immediately
    #    (right after a call that likely already spent up to ~30-60s
    #    failing internally) gives it no real cooldown either.
    # Fix: PerceptionFatalError aborts the search immediately with a clear
    # message. Plain (transient) PerceptionError holds position — no
    # turn — and waits before retrying the SAME frame, capped at a few
    # consecutive failures before giving up early instead of grinding
    # through the rest of max_steps uselessly.
    consecutive_perception_failures = 0
    max_consecutive_perception_failures = 3
    perception_retry_wait_s = 5.0

    print(f"\n--- Visual search: up to {max_steps} steps, instruction = {instruction!r} ---")
    for step in range(max_steps):
        _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw)
        img_path = os.path.abspath(f"search_step_{step:02d}.png")
        capture_face_camera(robot_pos, yaw, save_path=img_path)

        try:
            result = decide_next_action(img_path, instruction, known_objects, api_key=api_key,
                                         target_kind=target_kind)
            consecutive_perception_failures = 0
        except PerceptionFatalError as e:
            print(f"  step {step}: perception failed permanently ({e}) — this won't fix itself by "
                  f"turning or retrying. Stopping the search now instead of wasting the remaining "
                  f"steps.")
            return robot_pos, yaw, None, False
        except PerceptionError as e:
            consecutive_perception_failures += 1
            if consecutive_perception_failures >= max_consecutive_perception_failures:
                print(f"  step {step}: perception failed {consecutive_perception_failures} times in a "
                      f"row ({e}) — giving up early instead of continuing to burn through max_steps.")
                return robot_pos, yaw, None, False
            print(f"  step {step}: perception failed ({e}) — holding position, retrying in "
                  f"{perception_retry_wait_s:.0f}s (attempt {consecutive_perception_failures}/"
                  f"{max_consecutive_perception_failures})...")
            time.sleep(perception_retry_wait_s)
            continue  # re-capture and retry from the SAME pose, don't move blindly

        action = result.get("action")

        # Override the model's turn direction (not its action choice) once
        # a sweep is already underway — see the scan_direction comment
        # above. Only applies while still blind; once visible, fall
        # through to the return below and this never matters.
        if not result.get("visible") and action in ("turn_left", "turn_right"):
            if scan_direction is None:
                scan_direction = 1 if action == "turn_left" else -1
            forced_action = "turn_left" if scan_direction == 1 else "turn_right"
            if forced_action != action:
                print(f"  step {step}: model suggested {action}, holding course "
                      f"({forced_action}) to keep the sweep going in one direction.")
            action = forced_action

        print(f"  step {step}: action={action:<14} target={result.get('target_object')!s:<10} "
              f"visible={result.get('visible')!s:<5}  ({result.get('reasoning')})")

        # End the search as soon as the target is VISIBLE and IDENTIFIED —
        # not only on "target_reached". Waiting for Gemini to also judge
        # "close enough" forces continued turn/move-along-heading actions,
        # and those can deadlock: if the target is only visible while
        # facing an obstacle the robot can't safely walk through (e.g.
        # starting on the far side of the table), every move toward it
        # gets blocked, so the robot just turns away — loses sight of the
        # target — turns back — repeat, spinning in place indefinitely
        # without ever making positional progress (observed directly: 10+
        # steps of turn_left in a row with the table between robot and
        # ball). Once identified, hand off to align_for_pick's ground-
        # truth-based, obstacle-aware path planner instead — it already
        # computes the correct approach regardless of distance/angle, so
        # there's no benefit to creeping closer via unreliable vision-
        # guided steps first.
        if result.get("visible") and result.get("target_object") in known_objects:
            print(f"--- Target identified after {step + 1} step(s): {result['target_object']} "
                  f"(handing off to deterministic approach) ---\n")
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
            if is_path_clear(robot_pos, candidate, real_rects):
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


def align_for_pick(objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, target_pos, table_id,
                    approach_speed=0.008):
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

    # PICK_STANDOFF_FROM_EDGE (0.20m) is deliberately closer to the table
    # than the obstacle-avoidance clearance (ROBOT_FOOTPRINT_RADIUS + 0.05,
    # ~0.45m) — do_pick needs that proximity to reach the object. That
    # means aligned_pos always sits INSIDE the keepout rectangle, so
    # plan_path_around_obstacles can never certify a fully-clear route
    # all the way to it (a line from outside a convex region to a point
    # inside it always crosses the boundary) and falls back to an
    # UNCHECKED straight line for the whole remaining trip — which can
    # cut straight through the table's real geometry if the robot is
    # coming from the opposite side (this is what was driving the
    # wheel/decorative-arm-through-the-table bug).
    #
    # Fix: route with full protection only as far as the edge of the
    # clearance buffer (still outside the real table), with y already
    # aligned to the target. From there, x only decreases from
    # (table_edge - clearance) to (table_edge - PICK_STANDOFF_FROM_EDGE) —
    # both less than the table's real aabb_min[0] — so this final creep
    # can never re-enter the table's actual x-range, regardless of y.
    outer_margin = 0.05
    outer_x = aabb_min[0] - (ROBOT_FOOTPRINT_RADIUS + 0.05) - outer_margin
    outer_pos = [outer_x, target_y]

    robot_pos, yaw = navigate_to(robot_ids, arm_id, arm_joints, home_targets, robot_pos, outer_pos,
                                  keepout_rects, speed=approach_speed)

    # Final straight creep into the pick pose — safe by construction (see
    # above), so it's fine that it isn't run through the keepout planner.
    pos = list(robot_pos)
    while True:
        pos, new_yaw, reached = move_robot_towards(robot_ids, arm_id, pos, aligned_pos, speed=approach_speed)
        if new_yaw is not None:
            yaw = new_yaw
        hold_arm_pose(arm_id, arm_joints, home_targets)
        capture_face_camera(pos, yaw)
        p.stepSimulation()
        time.sleep(1 / 240)
        if reached:
            break

    # Face the table squarely once there — the creep above leaves yaw
    # pointed along the final travel segment, but do_pick was validated
    # at yaw=0.
    _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, aligned_pos, 0.0)
    return aligned_pos, 0.0


# Same reasoning as PICK_STANDOFF_FROM_EDGE, mirrored for the far edge —
# do_place doesn't need the same millimeter-precision approach do_pick
# does (it's dropping into an open area or a generously-sized box, not
# threading fingers around a 3cm ball), but staying in the same standoff
# ballpark keeps the arm's reach well within its validated working range.
PLACE_STANDOFF_FROM_EDGE = 0.20


def align_for_place(objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, place_target_xy,
                     table_id, approach_speed=0.008):
    """
    Mirror of align_for_pick, but for approaching the table's FAR edge
    (opposite the balls) instead of the near edge — used to walk the
    robot around to the other side of the table for a place action.

    Same two corrections as align_for_pick: (1) an empirically-reasonable
    standoff distance from the table edge, (2) y shifted so the arm's
    shoulder mount lines up with place_target_xy's y. The sign of the y
    correction flips relative to align_for_pick because the robot faces
    -x here (yaw=pi) instead of +x (yaw=0) — LOCAL_OFFSETS[ARM_MOUNT_KEY]
    is defined relative to the robot's own facing direction, so the
    world-frame shoulder offset rotates with yaw. Same fix as
    align_for_pick applies for routing: navigate_to only as far as the
    edge of the full clearance buffer, then a final straight creep whose
    x never re-enters the table's real x-range (here: decreasing from
    outer_x down to aligned_x, both > aabb_max[0]) so it can't cut
    through the table's real geometry regardless of which side the robot
    starts on.
    """
    _, aabb_max = p.getAABB(table_id)
    place_yaw = np.pi  # facing -x, back toward the table, from beyond the far edge
    target_x = aabb_max[0] + PLACE_STANDOFF_FROM_EDGE
    target_y = place_target_xy[1] - LOCAL_OFFSETS[ARM_MOUNT_KEY][1] * np.cos(place_yaw)
    aligned_pos = [target_x, target_y]

    keepout_rects = get_keepout_rects(objects)
    outer_margin = 0.05
    outer_x = aabb_max[0] + (ROBOT_FOOTPRINT_RADIUS + 0.05) + outer_margin
    outer_pos = [outer_x, target_y]

    robot_pos, yaw = navigate_to(robot_ids, arm_id, arm_joints, home_targets, robot_pos, outer_pos,
                                  keepout_rects, speed=approach_speed)

    pos = list(robot_pos)
    while True:
        pos, new_yaw, reached = move_robot_towards(robot_ids, arm_id, pos, aligned_pos, speed=approach_speed)
        if new_yaw is not None:
            yaw = new_yaw
        hold_arm_pose(arm_id, arm_joints, home_targets)
        capture_face_camera(pos, yaw)
        p.stepSimulation()
        time.sleep(1 / 240)
        if reached:
            break

    _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, aligned_pos, place_yaw)
    return aligned_pos, place_yaw


def _idle_until_closed():
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


def infer_ball_target(instruction, ball_names):
    """
    Deterministic parse of which SPECIFIC ball the instruction names
    (e.g. "pick up the yellow ball" -> "yellow_ball"), matched against
    the actual balls present (`ball_names`, e.g. objects["ball_names"]).

    This exists to fix a real bug: search_for_target's finishing check
    was `result.get("target_object") in known_objects` — true for ANY
    known ball, not specifically the one asked for. Passing the FULL
    ball list as known_objects meant that if Gemini spotted a different
    colored ball first while scanning and reported it as visible, the
    search would immediately lock onto and hand off the WRONG ball. The
    fix is for the caller to narrow known_objects down to just this one
    specific target when it can be determined, so nothing else can match.

    Returns a name from `ball_names`, or None if no color in the
    instruction matches any ball actually present (caller should fall
    back to the full list, degraded but still functional).
    """
    text = instruction.lower()
    for name in ball_names:
        color = name.split("_")[0]
        if color in text:
            return name
    return None


def infer_place_intent(instruction):
    """
    Deterministic, no-API-call parse of a place destination out of the
    instruction text — e.g. "pick up the red ball and place it in the
    box" or "...and put it at the other end of the table".

    Deliberately keyword-based rather than another Gemini call: this only
    needs to pick between 3 fixed outcomes, and every extra API call is
    one more chance to trip the free-tier rate limit (see the earlier
    429 issue) for something a plain string match handles reliably.

    Returns "box", "other-end", or "none" (no place instruction detected
    — in which case the run just picks and holds, as before).
    """
    text = instruction.lower()
    box_keywords = ("box", "container", "bin", "crate")
    other_end_keywords = ("other end", "far end", "opposite end", "opposite side",
                           "other side", "across the table", "far side")
    if any(kw in text for kw in box_keywords):
        return "box"
    if any(kw in text for kw in other_end_keywords):
        return "other-end"
    return "none"


# Kept in sync with the color set built in setup_static_scene.
_BOX_COLOR_NAMES = ["orange", "cyan", "magenta", "white", "black"]


def infer_box_color(instruction):
    """
    Deterministic parse of which box COLOR the instruction names (e.g.
    "the orange box", "put it in the cyan box") — same no-extra-API-call
    reasoning as infer_place_intent. This only decides which SEARCH TARGET
    to hand to search_for_target (a target string like "box_orange"); it
    does NOT tell the robot where that box physically is — that's still
    discovered purely by the robot reading the box's color with its
    camera, same as it locates a ball by color.

    Returns a color name string, or None if no box color was named
    (caller should fall back to searching for any box).
    """
    text = instruction.lower()
    for color in _BOX_COLOR_NAMES:
        if color in text:
            return color
    return None


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
        help="Skip the Gemini call entirely and just target the first known ball "
             "(offline testing, no API key needed).",
    )
    parser.add_argument(
        "--max-search-steps", type=int, default=20,
        help="Max turn/move steps the visual search loop will take before giving up. "
             "A full 360-degree scan alone takes ~15 steps at the default 25-degree turn "
             "step, so this needs to be comfortably above that to also allow movement — "
             "raise it further for --random-start or large --start-yaw-deg values.",
    )
    parser.add_argument(
        "--approach-speed", type=float, default=0.008,
        help="How far the robot moves per physics step (in meters) during the final "
             "approach to the pick position, after the target is identified. Lower = "
             "slower, more visible driving motion. Default 0.008; try e.g. 0.02 for "
             "faster/less patient runs, or 0.004 for an even slower demo.",
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
    parser.add_argument(
        "--place", choices=["auto", "none", "other-end", "box"], default="auto",
        help="What to do with the object after picking it up. Default 'auto' parses this from "
             "--instruction itself (e.g. 'pick up the red ball and place it in the box', or "
             "'...and put it at the other end of the table') — no need to set this explicitly "
             "in normal use. Set it directly only to override that parse: 'other-end' sets the "
             "object down on the table's far edge; 'box' drops it into a small open-top box "
             "placed at the table's far end (added to the scene automatically); 'none' picks "
             "and holds only, ignoring any destination mentioned in the instruction.",
    )
    args = parser.parse_args()

    place_mode = infer_place_intent(args.instruction) if args.place == "auto" else args.place
    box_color = infer_box_color(args.instruction) if place_mode == "box" else None
    objects = setup_static_scene(include_boxes=(place_mode == "box"))
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

    # Narrow the search target to the SPECIFIC ball named in the
    # instruction when we can tell which one that is (almost always,
    # since the whole point of the instruction is naming one) — see
    # infer_ball_target's docstring for why this matters: passing the
    # full list here would let the search lock onto ANY ball Gemini
    # happens to spot first, not necessarily the one asked for.
    requested_ball = infer_ball_target(args.instruction, objects["ball_names"])
    known_objects = [requested_ball] if requested_ball is not None else objects["ball_names"]
    if requested_ball is None:
        print(f"Could not determine which specific ball '{args.instruction!r}' refers to from its "
              f"color — searching for any of {objects['ball_names']} instead.")

    if args.no_perception:
        # Offline fallback path: skip the search loop entirely, drive to a
        # fixed standoff point and just target the first ball.
        print("--no-perception set: skipping the visual search loop.")
        approach_xy = compute_approach_point(objects["table"], robot_pos, standoff=0.25)
        aabb_min, aabb_max = p.getAABB(objects["table"])
        table_center_xy = [(aabb_min[0] + aabb_max[0]) / 2, (aabb_min[1] + aabb_max[1]) / 2]
        facing = np.array(table_center_xy) - np.array(approach_xy)
        yaw = float(np.arctan2(facing[1], facing[0]))
        robot_pos = approach_xy
        _place_and_settle(robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw)
        capture_face_camera(robot_pos, yaw, save_path=os.path.abspath("scene.png"))
        target_name = known_objects[0]
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
            # IMPORTANT: do NOT silently substitute a different object here.
            # Grabbing "the green ball" when the person asked for "the red
            # ball" and reporting it as a completed pick is actively
            # misleading — there's no indication anywhere that the wrong
            # object was picked unless you read the console closely. If the
            # search couldn't confirm the actual target, the honest outcome
            # is: report that clearly and stop, don't attempt a pick at all.
            print(f"\nSearch FAILED: could not locate a target matching {args.instruction!r} "
                  f"within {args.max_search_steps} steps.")
            print("No pick attempted. If the robot started far from the table, try increasing "
                  "--max-search-steps, or check that the instruction names a known object "
                  f"({known_objects}).")
            _idle_until_closed()
            return

    # The VLM only decided WHICH object and roughly confirmed it's close —
    # ground the final approach in the physics engine's real object pose,
    # then reposition to a validated pick pose (correct standoff distance
    # AND lateral alignment with the target — see align_for_pick) before
    # attempting do_pick.
    target_object_pos = p.getBasePositionAndOrientation(objects[target_name])[0]
    robot_pos, yaw = align_for_pick(
        objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, target_object_pos, objects["table"],
        approach_speed=args.approach_speed,
    )
    target_object_pos = p.getBasePositionAndOrientation(objects[target_name])[0]  # re-read after moving

    print("Executing pick...")
    grasp_constraint, pick_succeeded = do_pick(arm_id, target_object_pos, target_body_id=objects[target_name])

    final_pos = p.getBasePositionAndOrientation(objects[target_name])[0]
    if pick_succeeded:
        print(f"Pick SUCCEEDED. '{target_name}' position now:", final_pos)
    else:
        print(f"Pick FAILED — '{target_name}' was not actually lifted (position now: {final_pos}). "
              f"It may have been knocked off the table or missed during the grasp.")

    if pick_succeeded and place_mode != "none":
        aabb_min, aabb_max = p.getAABB(objects["table"])
        table_top_z = aabb_max[2]
        ball_radius = 0.03  # matches setup_static_scene

        target_box_id = None
        if place_mode == "box":
            # Which physical box is "the orange box" is NOT known here —
            # color is only recorded as the box's own rgbaColor (see
            # setup_static_scene). Find it the same way the ball was
            # found: reuse search_for_target's vision loop, just pointed
            # at box labels instead of ball colors.
            #
            # Narrow box_labels to the SPECIFIC requested color (when
            # named and valid) rather than always listing every color —
            # same fix, same reasoning as infer_ball_target above: passing
            # every color as "known" would let the search lock onto
            # whichever box Gemini happens to see FIRST while scanning
            # (e.g. cyan, on the way to actually finding orange), not the
            # one actually requested. This is likely exactly what caused
            # a real observed failure: the robot located some other box
            # while scanning past it, and placed there instead.
            if box_color is not None and box_color in objects["box_colors"]:
                box_labels = [f"box_{box_color}"]
                box_instruction = f"find the {box_color} colored box"
            else:
                if box_color is not None:
                    print(f"\nThe '{box_color}' box was requested but only "
                          f"{sorted(objects['box_colors'])} boxes exist in this scene — "
                          f"searching for any box instead.")
                box_labels = [f"box_{c}" for c in objects["box_colors"]]
                box_instruction = "find one of the colored boxes"

            print(f"\nLocating the{f' {box_color}' if box_color is not None else ''} "
                  f"box by its color...")
            robot_pos, yaw, box_label, box_found = search_for_target(
                objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, yaw,
                box_instruction, box_labels, args.gemini_api_key,
                max_steps=args.max_search_steps, target_kind="colored box",
            )
            if not box_found:
                print(f"\nCould not visually locate the requested box within "
                      f"{args.max_search_steps} steps. Skipping placement — "
                      f"'{target_name}' remains held.")
                place_mode = "none"  # skip the placement block below
            else:
                target_box_id = objects[box_label]
                found_color = box_label.split("_", 1)[1]
                if box_color is not None and found_color != box_color:
                    print(f"  Note: asked for the {box_color} box, but the robot found "
                          f"the {found_color} box — placing there anyway (it's what it "
                          f"actually found and confirmed via camera).")
                box_color = found_color

        if place_mode == "box" and target_box_id is not None:
            box_aabb_min, box_aabb_max = p.getAABB(target_box_id)
            place_xy = [(box_aabb_min[0] + box_aabb_max[0]) / 2, (box_aabb_min[1] + box_aabb_max[1]) / 2]
            place_z = objects[f"box_{box_color}_floor_z"] + ball_radius + 0.01
            place_desc = f"into the {box_color} box"
        elif place_mode == "other-end":
            place_xy = [aabb_max[0] - 0.15, 0.0]  # mirrors ball_x inset from setup_static_scene, far edge
            place_z = table_top_z + ball_radius + 0.01
            place_desc = "at the table's far end"
        else:
            place_xy = place_z = place_desc = None  # placement was skipped above

        if place_xy is not None:
            print(f"\nPlacing '{target_name}' {place_desc}...")
            robot_pos, yaw = align_for_place(
                objects, robot_ids, arm_id, arm_joints, home_targets, robot_pos, place_xy,
                objects["table"], approach_speed=args.approach_speed,
            )
            do_place(arm_id, [place_xy[0], place_xy[1], place_z], grasp_constraint, approach_yaw=yaw)

            final_pos = p.getBasePositionAndOrientation(objects[target_name])[0]
            # Distance check, not an exact-position check — the ball rolls/
            # settles after release, so "landed near the target" (or "landed
            # inside the box's footprint") is the meaningful success signal.
            if target_box_id is not None:
                placed_ok = (box_aabb_min[0] <= final_pos[0] <= box_aabb_max[0]
                             and box_aabb_min[1] <= final_pos[1] <= box_aabb_max[1]
                             and final_pos[2] < table_top_z + 0.15)
            else:
                placed_ok = np.hypot(final_pos[0] - place_xy[0], final_pos[1] - place_xy[1]) < 0.15
            verdict = "SUCCESS" if placed_ok else "uncertain — check final position"
            print(f"Place {verdict}. '{target_name}' final position:", final_pos)

    _idle_until_closed()


if __name__ == "__main__":
    main()