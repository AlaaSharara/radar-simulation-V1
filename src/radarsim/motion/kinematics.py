"""17-joint human forward kinematics, 14-segment capsule proxies, and full-body collision checks.

All quantities use SI units (metres, radians, seconds) and the world coordinate
convention from ``INTERFACES.md`` (+X right, +Y forward into room, +Z up).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Collection, Mapping, Sequence

import numpy as np

from radarsim.motion.types import BodyCapsule, ContactEvent, JOINT_NAMES, Vec3
from radarsim.scene import OrientedBox, Scene


@dataclass(frozen=True)
class PoseState:
    """Smoothly interpolatable parametric human pose."""

    root_x_m: float
    root_y_m: float
    root_yaw_rad: float
    pelvis_z_m: float
    pelvis_pitch_rad: float = 0.0
    pelvis_roll_rad: float = 0.0
    torso_pitch_rad: float = 0.0
    torso_roll_rad: float = 0.0
    head_pitch_rad: float = 0.0
    head_yaw_rad: float = 0.0
    left_hip_pitch_rad: float = 0.0
    left_knee_pitch_rad: float = 0.0
    right_hip_pitch_rad: float = 0.0
    right_knee_pitch_rad: float = 0.0
    left_shoulder_pitch_rad: float = 0.0
    left_shoulder_abduction_rad: float = 0.08
    left_elbow_pitch_rad: float = 0.12
    right_shoulder_pitch_rad: float = 0.0
    right_shoulder_abduction_rad: float = 0.08
    right_elbow_pitch_rad: float = 0.12
    leg_follow_pelvis: float = 0.0


def standing_pose(
    x_m: float,
    y_m: float,
    yaw_rad: float,
    height_m: float,
) -> PoseState:
    return PoseState(
        root_x_m=float(x_m),
        root_y_m=float(y_m),
        root_yaw_rad=float(yaw_rad),
        pelvis_z_m=0.530 * float(height_m),
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def lerp_angle(a: float, b: float, alpha: float) -> float:
    diff = _wrap_angle(b - a)
    return _wrap_angle(a + alpha * diff)


def smoothstep(u: float) -> float:
    t = max(0.0, min(1.0, u))
    return t * t * (3.0 - 2.0 * t)


def interpolate_pose(a: PoseState, b: PoseState, alpha: float) -> PoseState:
    t = max(0.0, min(1.0, float(alpha)))
    inv = 1.0 - t
    return PoseState(
        root_x_m=inv * a.root_x_m + t * b.root_x_m,
        root_y_m=inv * a.root_y_m + t * b.root_y_m,
        root_yaw_rad=lerp_angle(a.root_yaw_rad, b.root_yaw_rad, t),
        pelvis_z_m=inv * a.pelvis_z_m + t * b.pelvis_z_m,
        pelvis_pitch_rad=inv * a.pelvis_pitch_rad + t * b.pelvis_pitch_rad,
        pelvis_roll_rad=inv * a.pelvis_roll_rad + t * b.pelvis_roll_rad,
        torso_pitch_rad=inv * a.torso_pitch_rad + t * b.torso_pitch_rad,
        torso_roll_rad=inv * a.torso_roll_rad + t * b.torso_roll_rad,
        head_pitch_rad=inv * a.head_pitch_rad + t * b.head_pitch_rad,
        head_yaw_rad=inv * a.head_yaw_rad + t * b.head_yaw_rad,
        left_hip_pitch_rad=inv * a.left_hip_pitch_rad + t * b.left_hip_pitch_rad,
        left_knee_pitch_rad=inv * a.left_knee_pitch_rad + t * b.left_knee_pitch_rad,
        right_hip_pitch_rad=inv * a.right_hip_pitch_rad + t * b.right_hip_pitch_rad,
        right_knee_pitch_rad=inv * a.right_knee_pitch_rad + t * b.right_knee_pitch_rad,
        left_shoulder_pitch_rad=inv * a.left_shoulder_pitch_rad + t * b.left_shoulder_pitch_rad,
        left_shoulder_abduction_rad=inv * a.left_shoulder_abduction_rad + t * b.left_shoulder_abduction_rad,
        left_elbow_pitch_rad=inv * a.left_elbow_pitch_rad + t * b.left_elbow_pitch_rad,
        right_shoulder_pitch_rad=inv * a.right_shoulder_pitch_rad + t * b.right_shoulder_pitch_rad,
        right_shoulder_abduction_rad=inv * a.right_shoulder_abduction_rad + t * b.right_shoulder_abduction_rad,
        right_elbow_pitch_rad=inv * a.right_elbow_pitch_rad + t * b.right_elbow_pitch_rad,
        leg_follow_pelvis=inv * a.leg_follow_pelvis + t * b.leg_follow_pelvis,
    )


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def evaluate_joints(
    pose: PoseState,
    *,
    height_m: float,
    body_scale: float,
    floor_z_m: float = 0.0,
) -> dict[str, Vec3]:
    """Compute world-frame 3D coordinates [m] for all 17 ``INTERFACES.md`` joints.

    Coordinate convention: at ``yaw = 0``, actor faces ``+Y``, actor's right is
    ``+X``, and up is ``+Z``. Positive pitch tilts upper body forward toward ``+Y``.
    """
    h = float(height_m)
    s = (h / 1.78) * float(body_scale)

    # Segment lengths proportional to height
    l_spine = 0.105 * h
    l_chest = 0.140 * h
    l_neck = 0.070 * h
    l_head = 0.080 * h
    hip_half_w = 0.095 * s
    sh_half_w = 0.195 * s
    l_uarm = 0.178 * h
    l_farm = 0.155 * h
    l_thigh = 0.235 * h
    l_shin = 0.230 * h

    # Root yaw rotation: local +Y -> (-sin yaw, cos yaw, 0), local +X -> (cos yaw, sin yaw, 0)
    r_yaw = _rot_z(pose.root_yaw_rad)
    # Positive pitch tilts local +Z toward local +Y -> rotation around local +X by -pitch
    r_pelvis_local = _rot_y(pose.pelvis_roll_rad) @ _rot_x(-pose.pelvis_pitch_rad)
    r_pelvis = r_yaw @ r_pelvis_local

    r_torso_local = _rot_y(pose.torso_roll_rad) @ _rot_x(-pose.torso_pitch_rad)
    r_upper = r_pelvis @ r_torso_local
    r_head = r_upper @ _rot_z(pose.head_yaw_rad) @ _rot_x(-pose.head_pitch_rad)

    pelvis = np.array([pose.root_x_m, pose.root_y_m, pose.pelvis_z_m], dtype=np.float64)
    spine = pelvis + r_pelvis @ np.array([0.0, 0.0, l_spine])
    chest = spine + r_upper @ np.array([0.0, 0.0, l_chest])
    neck = chest + r_upper @ np.array([0.0, 0.0, l_neck])
    head = neck + r_head @ np.array([0.0, 0.01 * h, l_head])

    # Shoulders & Arms
    left_sh = chest + r_upper @ np.array([-sh_half_w, 0.0, 0.030 * h])
    right_sh = chest + r_upper @ np.array([sh_half_w, 0.0, 0.030 * h])

    def _arm_chain(
        sh_pos: np.ndarray,
        side_sign: float,  # -1 left, +1 right
        sh_pitch: float,
        sh_abd: float,
        el_pitch: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Pitch > 0 swings hanging arm forward (+Y) -> rotate (0,0,-1) around +X by +sh_pitch
        r_sh = r_upper @ _rot_x(sh_pitch) @ _rot_y(side_sign * sh_abd)
        elbow = sh_pos + r_sh @ np.array([0.0, 0.0, -l_uarm])
        r_el = r_sh @ _rot_x(el_pitch)
        wrist = elbow + r_el @ np.array([0.0, 0.0, -l_farm])
        return elbow, wrist

    left_el, left_wr = _arm_chain(
        left_sh,
        -1.0,
        pose.left_shoulder_pitch_rad,
        pose.left_shoulder_abduction_rad,
        pose.left_elbow_pitch_rad,
    )
    right_el, right_wr = _arm_chain(
        right_sh,
        +1.0,
        pose.right_shoulder_pitch_rad,
        pose.right_shoulder_abduction_rad,
        pose.right_elbow_pitch_rad,
    )

    # Legs (blend pelvis pitch/roll onto leg base frame via leg_follow_pelvis)
    r_leg_base = r_yaw @ _rot_y(pose.leg_follow_pelvis * pose.pelvis_roll_rad) @ _rot_x(
        -pose.leg_follow_pelvis * pose.pelvis_pitch_rad
    )
    left_hip = pelvis + r_pelvis @ np.array([-hip_half_w, 0.0, -0.015 * h])
    right_hip = pelvis + r_pelvis @ np.array([hip_half_w, 0.0, -0.015 * h])

    def _leg_chain(
        hip_pos: np.ndarray,
        hip_pitch: float,
        knee_pitch: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Hip pitch > 0 swings thigh forward (+Y); knee pitch > 0 bends shin backward (-Y)
        r_thigh = r_leg_base @ _rot_x(hip_pitch)
        knee = hip_pos + r_thigh @ np.array([0.0, 0.0, -l_thigh])
        r_shin = r_thigh @ _rot_x(-knee_pitch)
        ankle = knee + r_shin @ np.array([0.0, 0.0, -l_shin])
        return knee, ankle

    left_kn, left_an = _leg_chain(left_hip, pose.left_hip_pitch_rad, pose.left_knee_pitch_rad)
    right_kn, right_an = _leg_chain(right_hip, pose.right_hip_pitch_rad, pose.right_knee_pitch_rad)

    raw_joints: dict[str, np.ndarray] = {
        "pelvis": pelvis,
        "spine": spine,
        "chest": chest,
        "neck": neck,
        "head": head,
        "left_shoulder": left_sh,
        "left_elbow": left_el,
        "left_wrist": left_wr,
        "right_shoulder": right_sh,
        "right_elbow": right_el,
        "right_wrist": right_wr,
        "left_hip": left_hip,
        "left_knee": left_kn,
        "left_ankle": left_an,
        "right_hip": right_hip,
        "right_knee": right_kn,
        "right_ankle": right_an,
    }

    # Contact-aware floor clamping so no body segment penetrates below floor_z_m
    joint_clearance = {
        "left_ankle": 0.050,
        "right_ankle": 0.050,
        "left_knee": 0.060,
        "right_knee": 0.060,
        "left_wrist": 0.045,
        "right_wrist": 0.045,
        "left_elbow": 0.050,
        "right_elbow": 0.050,
        "head": 0.095,
        "pelvis": 0.110,
        "spine": 0.115,
        "chest": 0.120,
    }
    lift = 0.0
    for jname, pos in raw_joints.items():
        min_allowed = floor_z_m + joint_clearance.get(jname, 0.055)
        deficit = min_allowed - float(pos[2])
        if deficit > lift:
            lift = deficit

    if lift > 0.0:
        shift = np.array([0.0, 0.0, lift], dtype=np.float64)
        for jname in raw_joints:
            raw_joints[jname] = raw_joints[jname] + shift

    return {
        name: (float(raw_joints[name][0]), float(raw_joints[name][1]), float(raw_joints[name][2]))
        for name in JOINT_NAMES
    }


def build_body_capsules(
    joints_m: Mapping[str, Vec3],
    *,
    body_scale: float = 1.0,
    root_yaw_rad: float = 0.0,
    prev_joints_m: Mapping[str, Vec3] | None = None,
    dt_s: float = 0.01,
) -> list[BodyCapsule]:
    """Construct 14 anatomical collision and RF-scattering capsule proxies from 17 joints."""
    s = float(body_scale)
    fwd_x = -math.sin(root_yaw_rad)
    fwd_y = math.cos(root_yaw_rad)

    def _toe(ankle: Vec3) -> Vec3:
        return (ankle[0] + 0.10 * s * fwd_x, ankle[1] + 0.10 * s * fwd_y, max(0.035, ankle[2] - 0.015))

    specs: list[tuple[str, Vec3, Vec3, float, str, float, str, str]] = [
        ("head", joints_m["neck"], joints_m["head"], 0.095 * s, "fabric", -18.0, "neck", "head"),
        ("neck", joints_m["chest"], joints_m["neck"], 0.058 * s, "human_clothing", -22.0, "chest", "neck"),
        ("upper_torso", joints_m["spine"], joints_m["chest"], 0.145 * s, "human_clothing", -10.5, "spine", "chest"),
        ("lower_torso", joints_m["pelvis"], joints_m["spine"], 0.132 * s, "human_clothing", -12.0, "pelvis", "spine"),
        ("left_upper_arm", joints_m["left_shoulder"], joints_m["left_elbow"], 0.052 * s, "human_clothing", -19.0, "left_shoulder", "left_elbow"),
        ("left_forearm", joints_m["left_elbow"], joints_m["left_wrist"], 0.042 * s, "human_clothing", -20.0, "left_elbow", "left_wrist"),
        ("right_upper_arm", joints_m["right_shoulder"], joints_m["right_elbow"], 0.052 * s, "human_clothing", -19.0, "right_shoulder", "right_elbow"),
        ("right_forearm", joints_m["right_elbow"], joints_m["right_wrist"], 0.042 * s, "human_clothing", -20.0, "right_elbow", "right_wrist"),
        ("left_thigh", joints_m["left_hip"], joints_m["left_knee"], 0.078 * s, "human_clothing", -15.0, "left_hip", "left_knee"),
        ("left_shin", joints_m["left_knee"], joints_m["left_ankle"], 0.056 * s, "human_clothing", -17.0, "left_knee", "left_ankle"),
        ("right_thigh", joints_m["right_hip"], joints_m["right_knee"], 0.078 * s, "human_clothing", -15.0, "right_hip", "right_knee"),
        ("right_shin", joints_m["right_knee"], joints_m["right_ankle"], 0.056 * s, "human_clothing", -17.0, "right_knee", "right_ankle"),
        ("left_foot", joints_m["left_ankle"], _toe(joints_m["left_ankle"]), 0.040 * s, "human_clothing", -21.0, "left_ankle", "left_ankle"),
        ("right_foot", joints_m["right_ankle"], _toe(joints_m["right_ankle"]), 0.040 * s, "human_clothing", -21.0, "right_ankle", "right_ankle"),
    ]

    capsules: list[BodyCapsule] = []
    inv_dt = (1.0 / dt_s) if (prev_joints_m is not None and dt_s > 0.0) else 0.0
    for seg_id, a, b, rad, mat_id, rcs_db, ja, jb in specs:
        if prev_joints_m is not None and inv_dt > 0.0:
            pa = prev_joints_m[ja]
            pb = prev_joints_m[jb]
            ca_prev = (0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2]))
            ca_cur = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2]))
            vel = (
                (ca_cur[0] - ca_prev[0]) * inv_dt,
                (ca_cur[1] - ca_prev[1]) * inv_dt,
                (ca_cur[2] - ca_prev[2]) * inv_dt,
            )
        else:
            vel = (0.0, 0.0, 0.0)
        capsules.append(
            BodyCapsule(
                segment_id=seg_id,
                a_m=a,
                b_m=b,
                radius_m=rad,
                material_id=mat_id,
                rcs_prior_dbsm=rcs_db,
                velocity_mps=vel,
                metadata={"joint_a": ja, "joint_b": jb},
            )
        )
    return capsules


def _point_in_doorway_passage(x: float, y: float, scene: Scene, margin_m: float = 0.02) -> bool:
    for door in scene.doors():
        s0, s1 = door.span_along_wall_m
        if door.wall == "y_min" and (s0 - margin_m <= x <= s1 + margin_m) and y <= 0.35:
            return True
        if door.wall == "y_max" and (s0 - margin_m <= x <= s1 + margin_m) and y >= scene.depth_y_m - 0.35:
            return True
        if door.wall == "x_min" and (s0 - margin_m <= y <= s1 + margin_m) and x <= 0.35:
            return True
        if door.wall == "x_max" and (s0 - margin_m <= y <= s1 + margin_m) and x >= scene.width_x_m - 0.35:
            return True
    return False


def check_body_capsule_collisions(
    capsules: Sequence[BodyCapsule],
    scene: Scene,
    *,
    allowed_furniture_ids: Collection[str] = (),
    allow_doorway_exterior: bool = False,
    penetration_tolerance_m: float = 0.035,
) -> list[str]:
    """Check EVERY body capsule (not only joint centers) against floor, walls, and furniture.

    Returns a list of human-readable violation descriptions (empty if valid).
    """
    violations: list[str] = []
    allowed_set = set(allowed_furniture_ids)
    sample_ts = (0.0, 0.25, 0.5, 0.75, 1.0)

    for cap in capsules:
        eff_r = max(0.0, cap.radius_m - penetration_tolerance_m)
        for t in sample_ts:
            px = (1.0 - t) * cap.a_m[0] + t * cap.b_m[0]
            py = (1.0 - t) * cap.a_m[1] + t * cap.b_m[1]
            pz = (1.0 - t) * cap.a_m[2] + t * cap.b_m[2]

            if pz < -penetration_tolerance_m:
                violations.append(
                    f"Capsule '{cap.segment_id}' penetrates floor (z={pz:.3f} m)"
                )
                break

            if pz > scene.height_z_m + penetration_tolerance_m:
                violations.append(
                    f"Capsule '{cap.segment_id}' penetrates ceiling (z={pz:.3f} m)"
                )
                break

            in_room = (
                eff_r <= px <= scene.width_x_m - eff_r
                and eff_r <= py <= scene.depth_y_m - eff_r
            )
            if not in_room:
                if not (allow_doorway_exterior and _point_in_doorway_passage(px, py, scene)):
                    violations.append(
                        f"Capsule '{cap.segment_id}' clips room wall at ({px:.2f}, {py:.2f})"
                    )
                    break

            hit_furn = False
            for obj in scene.objects:
                if obj.object_id in allowed_set:
                    continue
                if obj.box.z_min_m - eff_r <= pz <= obj.box.z_max_m + eff_r:
                    if obj.box.distance_to_point_2d(px, py) < eff_r:
                        violations.append(
                            f"Capsule '{cap.segment_id}' collides with '{obj.object_id}' at ({px:.2f}, {py:.2f}, {pz:.2f})"
                        )
                        hit_furn = True
                        break
            if hit_furn:
                break

    return violations


def compute_contacts(
    joints_m: Mapping[str, Vec3],
    *,
    action_label: str,
    action_phase: str | None = None,
    contact_object_id: str | None = None,
    contact_surface_z_m: float = 0.0,
    intentional: bool = True,
    impact_speed_mps: float = 0.0,
) -> list[ContactEvent]:
    """Derive physically consistent floor and furniture contact events from current joints."""
    contacts: list[ContactEvent] = []

    # Floor foot contacts whenever ankles are near floor level
    for side in ("left", "right"):
        an = joints_m[f"{side}_ankle"]
        if an[2] <= 0.115:
            contacts.append(
                ContactEvent(
                    contact_id="floor",
                    surface_kind="floor",
                    body_part=f"{side}_foot",
                    contact_point_m=(an[0], an[1], 0.0),
                    intentional=True,
                    impact_speed_mps=0.0,
                )
            )

    # Knee floor contact (kneeling or fall)
    for side in ("left", "right"):
        kn = joints_m[f"{side}_knee"]
        if kn[2] <= 0.115:
            contacts.append(
                ContactEvent(
                    contact_id="floor",
                    surface_kind="floor",
                    body_part=f"{side}_knee",
                    contact_point_m=(kn[0], kn[1], 0.0),
                    intentional=intentional,
                    impact_speed_mps=impact_speed_mps if not intentional else 0.0,
                )
            )

    # Torso / pelvis / hip / shoulder / head floor contact during fall impact/settling
    if not intentional and action_phase in ("impact_contact", "settling", "optional_recovery"):
        for part, thresh_z in (
            ("pelvis", 0.26),
            ("chest", 0.26),
            ("head", 0.26),
            ("left_hip", 0.18),
            ("right_hip", 0.18),
            ("left_shoulder", 0.16),
            ("right_shoulder", 0.16),
        ):
            pt = joints_m[part]
            if pt[2] <= thresh_z:
                contacts.append(
                    ContactEvent(
                        contact_id="floor",
                        surface_kind="floor",
                        body_part=part,
                        contact_point_m=(pt[0], pt[1], 0.0),
                        intentional=False,
                        impact_speed_mps=impact_speed_mps if action_phase == "impact_contact" else 0.0,
                    )
                )

    # Intentional furniture contact (sitting / lying / sleeping on bed, sofa, or chair)
    if contact_object_id is not None:
        pelvis = joints_m["pelvis"]
        if abs(pelvis[2] - (contact_surface_z_m + 0.11)) <= 0.22:
            contacts.append(
                ContactEvent(
                    contact_id=contact_object_id,
                    surface_kind="furniture",
                    body_part="pelvis",
                    contact_point_m=(pelvis[0], pelvis[1], contact_surface_z_m),
                    intentional=True,
                    impact_speed_mps=0.0,
                )
            )
        chest = joints_m["chest"]
        if abs(chest[2] - (contact_surface_z_m + 0.12)) <= 0.22:
            contacts.append(
                ContactEvent(
                    contact_id=contact_object_id,
                    surface_kind="furniture",
                    body_part="torso",
                    contact_point_m=(chest[0], chest[1], contact_surface_z_m),
                    intentional=True,
                    impact_speed_mps=0.0,
                )
            )

    return contacts
