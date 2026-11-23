"""Deterministic 100 Hz human motion & scenario generator.

Supports ``normal_daily_routine``, ``fall_demo``, and ``random_mix`` modes from
``motion_config.yaml``, collision-aware A* navigation around furniture from the
normalized ``Scene``, intentional furniture contacts, hard-negative actions,
and 5 contact-aware multi-phase fall styles using independent seeded RNG streams.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

import numpy as np

from radarsim.config import MotionConfig
from radarsim.motion.avatar import AVATAR_ASSET_ID
from radarsim.motion.kinematics import (
    PoseState,
    build_body_capsules,
    check_body_capsule_collisions,
    compute_contacts,
    evaluate_joints,
    interpolate_pose,
    lerp_angle,
    smoothstep,
    standing_pose,
)
from radarsim.motion.navigation import AStarNavigator, FurnitureInteractionSpot, XY
from radarsim.motion.types import (
    ActionSegment,
    FALL_STYLES,
    MotionFrame,
    MotionSequence,
    Vec3,
)
from radarsim.rng import SeedManager
from radarsim.scene import Scene


def _yaw_from_delta(dx: float, dy: float, fallback: float = 0.0) -> float:
    if math.hypot(dx, dy) < 1e-6:
        return fallback
    # At yaw=0, forward is (0, +1); forward = (-sin yaw, cos yaw) -> yaw = atan2(-dx, dy)
    return math.atan2(-dx, dy)


def _walk_pose_at_phase(
    x: float,
    y: float,
    yaw: float,
    height_m: float,
    phase_rad: float,
    speed_mps: float,
) -> PoseState:
    stride_amp = min(0.34, max(0.14, 0.25 * speed_mps))
    s = math.sin(phase_rad)
    c = math.cos(phase_rad)
    l_hip = stride_amp * s
    r_hip = -stride_amp * s
    l_knee = max(0.0, 0.65 * stride_amp * (0.5 - 0.5 * c) + 0.10)
    r_knee = max(0.0, 0.65 * stride_amp * (0.5 + 0.5 * c) + 0.10)
    bob = 0.016 * abs(s)
    return PoseState(
        root_x_m=x,
        root_y_m=y,
        root_yaw_rad=yaw,
        pelvis_z_m=0.530 * height_m - bob,
        torso_pitch_rad=0.04 * min(1.2, speed_mps),
        left_hip_pitch_rad=l_hip,
        left_knee_pitch_rad=l_knee,
        right_hip_pitch_rad=r_hip,
        right_knee_pitch_rad=r_knee,
        left_shoulder_pitch_rad=-0.65 * stride_amp * s,
        right_shoulder_pitch_rad=0.65 * stride_amp * s,
        left_elbow_pitch_rad=0.25 + 0.10 * max(0.0, -s),
        right_elbow_pitch_rad=0.25 + 0.10 * max(0.0, s),
    )


def _fall_target_pose(
    x: float,
    y: float,
    yaw: float,
    height_m: float,
    fall_style: str,
    side_sign: float = 1.0,
) -> PoseState:
    """Final horizontal/floor-contact pose for each of the 5 configured fall styles.

    Limb angles keep arms and legs resting on/parallel to the floor plane rather
    than pointing down through the floor.
    """
    if fall_style == "fall_forward":
        return PoseState(
            root_x_m=x,
            root_y_m=y,
            root_yaw_rad=yaw,
            pelvis_z_m=0.120,
            pelvis_pitch_rad=0.49 * math.pi,
            torso_pitch_rad=0.0,
            head_pitch_rad=-0.10,
            left_hip_pitch_rad=-0.04,
            left_knee_pitch_rad=0.10,
            right_hip_pitch_rad=-0.02,
            right_knee_pitch_rad=0.12,
            left_shoulder_pitch_rad=-0.12,
            left_shoulder_abduction_rad=0.45,
            left_elbow_pitch_rad=0.10,
            right_shoulder_pitch_rad=-0.15,
            right_shoulder_abduction_rad=0.42,
            right_elbow_pitch_rad=0.12,
            leg_follow_pelvis=1.0,
        )
    if fall_style == "fall_backward":
        return PoseState(
            root_x_m=x,
            root_y_m=y,
            root_yaw_rad=yaw,
            pelvis_z_m=0.120,
            pelvis_pitch_rad=-0.49 * math.pi,
            torso_pitch_rad=0.0,
            head_pitch_rad=0.08,
            left_hip_pitch_rad=0.08,
            left_knee_pitch_rad=0.12,
            right_hip_pitch_rad=0.10,
            right_knee_pitch_rad=0.14,
            left_shoulder_pitch_rad=0.10,
            left_shoulder_abduction_rad=0.48,
            left_elbow_pitch_rad=0.15,
            right_shoulder_pitch_rad=0.12,
            right_shoulder_abduction_rad=0.52,
            right_elbow_pitch_rad=0.18,
            leg_follow_pelvis=1.0,
        )
    if fall_style == "fall_sideways":
        return PoseState(
            root_x_m=x,
            root_y_m=y,
            root_yaw_rad=yaw,
            pelvis_z_m=0.125,
            pelvis_roll_rad=side_sign * 0.49 * math.pi,
            torso_pitch_rad=0.06,
            left_hip_pitch_rad=0.22,
            left_knee_pitch_rad=0.28,
            right_hip_pitch_rad=0.14,
            right_knee_pitch_rad=0.20,
            left_shoulder_pitch_rad=0.18,
            left_shoulder_abduction_rad=0.12,
            left_elbow_pitch_rad=0.35,
            right_shoulder_pitch_rad=0.15,
            right_shoulder_abduction_rad=0.12,
            right_elbow_pitch_rad=0.30,
            leg_follow_pelvis=1.0,
        )
    if fall_style == "fall_slump":
        return PoseState(
            root_x_m=x,
            root_y_m=y,
            root_yaw_rad=yaw,
            pelvis_z_m=0.125,
            pelvis_pitch_rad=0.45 * math.pi,
            pelvis_roll_rad=side_sign * 0.18 * math.pi,
            torso_pitch_rad=0.05,
            left_hip_pitch_rad=0.05,
            left_knee_pitch_rad=0.25,
            right_hip_pitch_rad=0.08,
            right_knee_pitch_rad=0.30,
            left_shoulder_pitch_rad=-0.10,
            left_shoulder_abduction_rad=0.35,
            left_elbow_pitch_rad=0.15,
            right_shoulder_pitch_rad=-0.12,
            right_shoulder_abduction_rad=0.32,
            right_elbow_pitch_rad=0.18,
            leg_follow_pelvis=1.0,
        )
    # fall_slow_collapse
    return PoseState(
        root_x_m=x,
        root_y_m=y,
        root_yaw_rad=yaw,
        pelvis_z_m=0.120,
        pelvis_pitch_rad=0.47 * math.pi,
        pelvis_roll_rad=side_sign * 0.12 * math.pi,
        torso_pitch_rad=0.04,
        left_hip_pitch_rad=0.02,
        left_knee_pitch_rad=0.18,
        right_hip_pitch_rad=0.04,
        right_knee_pitch_rad=0.22,
        left_shoulder_pitch_rad=-0.10,
        left_shoulder_abduction_rad=0.38,
        left_elbow_pitch_rad=0.14,
        right_shoulder_pitch_rad=-0.12,
        right_shoulder_abduction_rad=0.35,
        right_elbow_pitch_rad=0.16,
        leg_follow_pelvis=1.0,
    )


def _find_safe_fall_spot(
    navigator: AStarNavigator,
    scene: Scene,
    current_xy: XY,
    rng_path: np.random.Generator,
    rng_fall: np.random.Generator,
    *,
    height_m: float,
    body_scale: float,
    fall_style: str,
    side_sign: float,
) -> tuple[XY, float]:
    """Find a navigable open-floor position and yaw where the fallen body has zero collisions."""
    candidates = navigator.free_cells_xy(min_wall_margin_m=0.45)
    if not candidates:
        candidates = navigator.free_cells_xy(min_wall_margin_m=0.20)
    if not candidates:
        return current_xy, 0.0

    # Shuffle deterministically using rng_path and score candidates with clear floor radius
    order = rng_path.permutation(len(candidates))
    yaw_trials = [float(rng_fall.uniform(-math.pi, math.pi)) + k * (math.pi / 4.0) for k in range(8)]

    for idx in order[:80]:
        cx, cy = candidates[int(idx)]
        # Prefer reachable points
        if navigator.plan_path(current_xy, (cx, cy)) is None:
            continue
        for yaw in yaw_trials:
            yaw_w = math.atan2(math.sin(yaw), math.cos(yaw))
            final_pose = _fall_target_pose(cx, cy, yaw_w, height_m, fall_style, side_sign=side_sign)
            joints = evaluate_joints(final_pose, height_m=height_m, body_scale=body_scale)
            caps = build_body_capsules(joints, body_scale=body_scale, root_yaw_rad=yaw_w)
            if not check_body_capsule_collisions(caps, scene):
                return (cx, cy), yaw_w

    # Fallback: test yaw orientations at current_xy
    for yaw in yaw_trials:
        yaw_w = math.atan2(math.sin(yaw), math.cos(yaw))
        final_pose = _fall_target_pose(current_xy[0], current_xy[1], yaw_w, height_m, fall_style, side_sign=side_sign)
        joints = evaluate_joints(final_pose, height_m=height_m, body_scale=body_scale)
        caps = build_body_capsules(joints, body_scale=body_scale, root_yaw_rad=yaw_w)
        if not check_body_capsule_collisions(caps, scene):
            return current_xy, yaw_w

    return current_xy, 0.0


class _MotionBuilder:
    """Accumulates 100 Hz ``MotionFrame`` objects with smooth transitions and capsule velocities."""

    def __init__(
        self,
        *,
        scene: Scene,
        sample_rate_hz: float,
        actor_id: str,
        avatar_asset_id: str,
        scenario_mode: str,
        height_m: float,
        body_scale: float,
        initial_pose: PoseState,
    ) -> None:
        self.scene = scene
        self.sample_rate_hz = float(sample_rate_hz)
        self.dt_s = 1.0 / self.sample_rate_hz
        self.actor_id = actor_id
        self.avatar_asset_id = avatar_asset_id
        self.scenario_mode = scenario_mode
        self.height_m = height_m
        self.body_scale = body_scale
        self.pose = initial_pose
        self.frames: list[MotionFrame] = []
        self.events: list[ActionSegment] = []
        self.fall_events: list[dict[str, Any]] = []
        self._prev_joints: dict[str, Vec3] | None = None

    def append_frame(
        self,
        pose: PoseState,
        *,
        action_label: str,
        action_phase: str | None = None,
        fall_event_id: str | None = None,
        fall_style: str | None = None,
        is_hard_negative: bool = False,
        contact_object_id: str | None = None,
        contact_surface_z_m: float = 0.0,
        intentional_contact: bool = True,
        impact_speed_mps: float = 0.0,
    ) -> MotionFrame:
        self.pose = pose
        frame_id = len(self.frames)
        sim_time_s = round(frame_id * self.dt_s, 6)
        joints = evaluate_joints(
            pose,
            height_m=self.height_m,
            body_scale=self.body_scale,
            floor_z_m=0.0,
        )
        capsules = build_body_capsules(
            joints,
            body_scale=self.body_scale,
            root_yaw_rad=pose.root_yaw_rad,
            prev_joints_m=self._prev_joints,
            dt_s=self.dt_s,
        )
        contacts = compute_contacts(
            joints,
            action_label=action_label,
            action_phase=action_phase,
            contact_object_id=contact_object_id,
            contact_surface_z_m=contact_surface_z_m,
            intentional=intentional_contact,
            impact_speed_mps=impact_speed_mps,
        )
        frame = MotionFrame(
            frame_id=frame_id,
            sim_time_s=sim_time_s,
            actor_id=self.actor_id,
            action_label=action_label,
            scenario_mode=self.scenario_mode,
            fall_event_id=fall_event_id,
            root_position_m=(pose.root_x_m, pose.root_y_m, 0.0),
            root_yaw_rad=pose.root_yaw_rad,
            joints_m=joints,
            body_capsules=capsules,
            contacts=contacts,
            avatar_asset_id=self.avatar_asset_id,
            action_phase=action_phase,
            is_hard_negative=is_hard_negative,
            fall_style=fall_style,
        )
        self.frames.append(frame)
        self._prev_joints = joints
        return frame

    def emit_transition(
        self,
        target_pose: PoseState,
        duration_s: float,
        *,
        action_label: str,
        action_phase: str | None = None,
        fall_event_id: str | None = None,
        fall_style: str | None = None,
        is_hard_negative: bool = False,
        contact_object_id: str | None = None,
        contact_surface_z_m: float = 0.0,
        intentional_contact: bool = True,
        ease_power: float | None = None,
        impact_speed_mps: float = 0.0,
    ) -> None:
        start_pose = self.pose
        n_steps = max(1, int(round(duration_s * self.sample_rate_hz)))
        for k in range(1, n_steps + 1):
            u = k / n_steps
            alpha = (u ** ease_power) if ease_power is not None else smoothstep(u)
            p = interpolate_pose(start_pose, target_pose, alpha)
            self.append_frame(
                p,
                action_label=action_label,
                action_phase=action_phase,
                fall_event_id=fall_event_id,
                fall_style=fall_style,
                is_hard_negative=is_hard_negative,
                contact_object_id=contact_object_id,
                contact_surface_z_m=contact_surface_z_m,
                intentional_contact=intentional_contact,
                impact_speed_mps=impact_speed_mps,
            )

    def emit_walk_along_path(
        self,
        waypoints: Sequence[XY],
        speed_mps: float,
        *,
        action_label: str = "walk",
    ) -> ActionSegment:
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        phase = 0.0
        stride_freq_hz = max(0.7, min(1.5, 0.95 * speed_mps))

        for seg_idx in range(len(waypoints) - 1):
            ax, ay = waypoints[seg_idx]
            bx, by = waypoints[seg_idx + 1]
            dist = math.hypot(bx - ax, by - ay)
            if dist < 1e-4:
                continue
            target_yaw = _yaw_from_delta(bx - ax, by - ay, fallback=self.pose.root_yaw_rad)
            # Smoothly align yaw before walking long segment
            yaw_diff = abs(math.atan2(math.sin(target_yaw - self.pose.root_yaw_rad), math.cos(target_yaw - self.pose.root_yaw_rad)))
            if yaw_diff > 0.25:
                turn_dur = min(0.45, max(0.12, yaw_diff / 2.2))
                self.emit_transition(
                    standing_pose(self.pose.root_x_m, self.pose.root_y_m, target_yaw, self.height_m),
                    turn_dur,
                    action_label=action_label,
                )

            dur = max(self.dt_s, dist / max(0.15, speed_mps))
            n_steps = max(1, int(round(dur * self.sample_rate_hz)))
            for k in range(1, n_steps + 1):
                u = k / n_steps
                px = (1.0 - u) * ax + u * bx
                py = (1.0 - u) * ay + u * by
                phase += 2.0 * math.pi * stride_freq_hz * self.dt_s
                cur_yaw = lerp_angle(self.pose.root_yaw_rad, target_yaw, min(1.0, 0.25))
                p = _walk_pose_at_phase(px, py, cur_yaw, self.height_m, phase, speed_mps)
                self.append_frame(p, action_label=action_label)

        # Return smoothly to neutral standing at destination
        end_stand = standing_pose(self.pose.root_x_m, self.pose.root_y_m, self.pose.root_yaw_rad, self.height_m)
        self.emit_transition(end_stand, 0.15, action_label=action_label)
        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label=action_label,
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.events.append(seg)
        return seg

    def emit_turn(self, target_yaw_rad: float, turn_rate_rad_s: float) -> ActionSegment:
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        diff = abs(math.atan2(math.sin(target_yaw_rad - self.pose.root_yaw_rad), math.cos(target_yaw_rad - self.pose.root_yaw_rad)))
        dur = max(0.25, diff / max(0.4, turn_rate_rad_s))
        target = standing_pose(self.pose.root_x_m, self.pose.root_y_m, target_yaw_rad, self.height_m)
        self.emit_transition(target, dur, action_label="turn")
        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label="turn",
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.events.append(seg)
        return seg

    def emit_pause(self, duration_s: float) -> ActionSegment:
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        base = standing_pose(self.pose.root_x_m, self.pose.root_y_m, self.pose.root_yaw_rad, self.height_m)
        n_steps = max(1, int(round(duration_s * self.sample_rate_hz)))
        for k in range(n_steps):
            t = k * self.dt_s
            breath = 0.012 * math.sin(2.0 * math.pi * 0.25 * t)
            sway = 0.008 * math.sin(2.0 * math.pi * 0.15 * t)
            p = replace(base, torso_pitch_rad=breath, torso_roll_rad=sway)
            self.append_frame(p, action_label="pause")
        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label="pause",
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self.events.append(seg)
        return seg

    def emit_bend_or_pickup(
        self,
        *,
        action_label: str = "bend",
        is_hard_negative: bool = False,
        hold_s: float = 0.45,
    ) -> ActionSegment:
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        stand = standing_pose(self.pose.root_x_m, self.pose.root_y_m, self.pose.root_yaw_rad, self.height_m)
        bent = PoseState(
            root_x_m=stand.root_x_m,
            root_y_m=stand.root_y_m,
            root_yaw_rad=stand.root_yaw_rad,
            pelvis_z_m=0.47 * self.height_m,
            torso_pitch_rad=1.18,
            head_pitch_rad=0.25,
            left_hip_pitch_rad=0.35,
            left_knee_pitch_rad=0.40,
            right_hip_pitch_rad=0.35,
            right_knee_pitch_rad=0.40,
            left_shoulder_pitch_rad=0.95,
            left_elbow_pitch_rad=0.20,
            right_shoulder_pitch_rad=1.40,
            right_elbow_pitch_rad=0.15,
        )
        down_dur = 0.45 if is_hard_negative else 0.65
        self.emit_transition(bent, down_dur, action_label=action_label, is_hard_negative=is_hard_negative)
        self.emit_transition(bent, hold_s, action_label=action_label, is_hard_negative=is_hard_negative)
        self.emit_transition(stand, 0.60, action_label=action_label, is_hard_negative=is_hard_negative)
        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label=action_label,
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
            is_hard_negative=is_hard_negative,
        )
        self.events.append(seg)
        return seg

    def emit_kneel(
        self,
        *,
        action_label: str = "kneel",
        is_hard_negative: bool = False,
        hold_s: float = 0.80,
    ) -> ActionSegment:
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        stand = standing_pose(self.pose.root_x_m, self.pose.root_y_m, self.pose.root_yaw_rad, self.height_m)
        kneeling = PoseState(
            root_x_m=stand.root_x_m,
            root_y_m=stand.root_y_m,
            root_yaw_rad=stand.root_yaw_rad,
            pelvis_z_m=0.29 * self.height_m,
            torso_pitch_rad=0.18,
            left_hip_pitch_rad=0.18,
            left_knee_pitch_rad=1.65,
            right_hip_pitch_rad=1.25,
            right_knee_pitch_rad=1.45,
            left_shoulder_pitch_rad=0.35,
            left_elbow_pitch_rad=0.55,
            right_shoulder_pitch_rad=0.45,
            right_elbow_pitch_rad=0.60,
        )
        self.emit_transition(kneeling, 0.70, action_label=action_label, is_hard_negative=is_hard_negative)
        self.emit_transition(kneeling, hold_s, action_label=action_label, is_hard_negative=is_hard_negative)
        self.emit_transition(stand, 0.75, action_label=action_label, is_hard_negative=is_hard_negative)
        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label=action_label,
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
            is_hard_negative=is_hard_negative,
        )
        self.events.append(seg)
        return seg

    def emit_sit_and_stand(
        self,
        spot: FurnitureInteractionSpot,
        sit_duration_s: float,
        *,
        sit_label: str = "sit",
        is_hard_negative: bool = False,
        quick: bool = False,
    ) -> tuple[ActionSegment, ActionSegment]:
        # Align with outward facing yaw before sitting
        self.emit_transition(
            standing_pose(spot.approach_xy_m[0], spot.approach_xy_m[1], spot.approach_yaw_rad, self.height_m),
            0.35,
            action_label=sit_label,
            is_hard_negative=is_hard_negative,
        )
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)

        seat_z = min(spot.surface_z_m, 0.56)
        seated = PoseState(
            root_x_m=spot.seat_or_lie_xy_m[0],
            root_y_m=spot.seat_or_lie_xy_m[1],
            root_yaw_rad=spot.seat_or_lie_yaw_rad,
            pelvis_z_m=seat_z + 0.11,
            torso_pitch_rad=-0.06,
            left_hip_pitch_rad=1.42,
            left_knee_pitch_rad=1.42,
            right_hip_pitch_rad=1.42,
            right_knee_pitch_rad=1.42,
            left_shoulder_pitch_rad=0.25,
            left_elbow_pitch_rad=0.95,
            right_shoulder_pitch_rad=0.25,
            right_elbow_pitch_rad=0.95,
        )
        descend_s = 0.42 if quick else 0.85
        self.emit_transition(
            seated,
            descend_s,
            action_label=sit_label,
            is_hard_negative=is_hard_negative,
            contact_object_id=spot.object_id,
            contact_surface_z_m=spot.surface_z_m,
            intentional_contact=True,
        )
        # Hold seated with subtle breathing
        n_hold = max(1, int(round(sit_duration_s * self.sample_rate_hz)))
        for k in range(n_hold):
            t = k * self.dt_s
            p = replace(seated, torso_pitch_rad=-0.06 + 0.015 * math.sin(2.0 * math.pi * 0.25 * t))
            self.append_frame(
                p,
                action_label=sit_label,
                is_hard_negative=is_hard_negative,
                contact_object_id=spot.object_id,
                contact_surface_z_m=spot.surface_z_m,
                intentional_contact=True,
            )
        sit_end = len(self.frames) - 1
        sit_seg = ActionSegment(
            action_label=sit_label,
            start_time_s=start_time,
            end_time_s=round(sit_end * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=sit_end,
            is_hard_negative=is_hard_negative,
            target_object_id=spot.object_id,
        )
        self.events.append(sit_seg)

        # Stand back up to approach spot
        stand_start = len(self.frames)
        stand_start_t = round(stand_start * self.dt_s, 6)
        upright = standing_pose(spot.approach_xy_m[0], spot.approach_xy_m[1], spot.approach_yaw_rad, self.height_m)
        self.emit_transition(
            upright,
            0.85,
            action_label="stand",
            contact_object_id=spot.object_id,
            contact_surface_z_m=spot.surface_z_m,
            intentional_contact=True,
        )
        stand_end = len(self.frames) - 1
        stand_seg = ActionSegment(
            action_label="stand",
            start_time_s=stand_start_t,
            end_time_s=round(stand_end * self.dt_s, 6),
            start_frame=stand_start,
            end_frame=stand_end,
            target_object_id=spot.object_id,
        )
        self.events.append(stand_seg)
        return sit_seg, stand_seg

    def emit_lie_or_sleep(
        self,
        spot: FurnitureInteractionSpot,
        lie_duration_s: float,
        *,
        action_label: str = "lie_down_intentionally",
        is_hard_negative: bool = True,
        with_roll: bool = False,
    ) -> ActionSegment:
        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        surf_z = spot.surface_z_m

        # Step 1: Sit on edge of bed/sofa (controlled intentional transition)
        seated = PoseState(
            root_x_m=spot.seat_or_lie_xy_m[0],
            root_y_m=spot.seat_or_lie_xy_m[1],
            root_yaw_rad=spot.seat_or_lie_yaw_rad,
            pelvis_z_m=surf_z + 0.12,
            left_hip_pitch_rad=1.25,
            left_knee_pitch_rad=1.25,
            right_hip_pitch_rad=1.25,
            right_knee_pitch_rad=1.25,
        )
        self.emit_transition(
            seated,
            0.75,
            action_label=action_label,
            is_hard_negative=is_hard_negative,
            contact_object_id=spot.object_id,
            contact_surface_z_m=surf_z,
            intentional_contact=True,
        )

        # Step 2: Recline smoothly onto bed/sofa top surface
        supine = PoseState(
            root_x_m=spot.seat_or_lie_xy_m[0],
            root_y_m=spot.seat_or_lie_xy_m[1],
            root_yaw_rad=spot.seat_or_lie_yaw_rad,
            pelvis_z_m=surf_z + 0.13,
            pelvis_pitch_rad=-0.48 * math.pi,
            left_hip_pitch_rad=0.10,
            left_knee_pitch_rad=0.15,
            right_hip_pitch_rad=0.10,
            right_knee_pitch_rad=0.15,
            left_shoulder_abduction_rad=0.22,
            right_shoulder_abduction_rad=0.22,
            left_elbow_pitch_rad=0.30,
            right_elbow_pitch_rad=0.30,
            leg_follow_pelvis=1.0,
        )
        self.emit_transition(
            supine,
            1.0,
            action_label=action_label,
            is_hard_negative=is_hard_negative,
            contact_object_id=spot.object_id,
            contact_surface_z_m=surf_z,
            intentional_contact=True,
        )

        # Step 3: Rest / sleep (optional gentle roll-over)
        n_hold = max(1, int(round(lie_duration_s * self.sample_rate_hz)))
        for k in range(n_hold):
            t = k * self.dt_s
            roll = 0.55 * math.sin(2.0 * math.pi * t / max(2.0, lie_duration_s)) if with_roll else 0.0
            breath = 0.015 * math.sin(2.0 * math.pi * 0.22 * t)
            p = replace(supine, pelvis_roll_rad=roll, torso_pitch_rad=breath)
            self.append_frame(
                p,
                action_label=action_label,
                is_hard_negative=is_hard_negative,
                contact_object_id=spot.object_id,
                contact_surface_z_m=surf_z,
                intentional_contact=True,
            )

        # Step 4: Controlled sit-up and return to standing at approach spot
        self.emit_transition(
            seated,
            0.90,
            action_label=action_label,
            is_hard_negative=is_hard_negative,
            contact_object_id=spot.object_id,
            contact_surface_z_m=surf_z,
            intentional_contact=True,
        )
        upright = standing_pose(spot.approach_xy_m[0], spot.approach_xy_m[1], spot.approach_yaw_rad, self.height_m)
        self.emit_transition(
            upright,
            0.80,
            action_label="stand",
            contact_object_id=spot.object_id,
            contact_surface_z_m=surf_z,
            intentional_contact=True,
        )

        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label=action_label,
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
            is_fall=False,
            is_hard_negative=is_hard_negative,
            target_object_id=spot.object_id,
        )
        self.events.append(seg)
        return seg

    def emit_fall(
        self,
        *,
        fall_style: str,
        fall_event_id: str,
        fall_yaw_rad: float,
        descent_duration_s: float,
        recover: bool,
        side_sign: float = 1.0,
    ) -> ActionSegment:
        """Emit all 5 contact-aware fall phases without pose teleportation."""
        if fall_style not in FALL_STYLES:
            raise ValueError(f"Unsupported fall style: {fall_style!r}")

        # Smoothly align to the safe fall yaw while standing
        stand = standing_pose(self.pose.root_x_m, self.pose.root_y_m, fall_yaw_rad, self.height_m)
        if abs(math.atan2(math.sin(fall_yaw_rad - self.pose.root_yaw_rad), math.cos(fall_yaw_rad - self.pose.root_yaw_rad))) > 0.15:
            self.emit_transition(stand, 0.30, action_label="turn")

        start_frame = len(self.frames)
        start_time = round(start_frame * self.dt_s, 6)
        phase_ts: dict[str, float] = {}

        final_pose = _fall_target_pose(
            stand.root_x_m, stand.root_y_m, fall_yaw_rad, self.height_m, fall_style, side_sign=side_sign
        )

        # Phase 1: loss_of_balance (destabilizing lean & knee buckle)
        phase_ts["loss_of_balance"] = round(len(self.frames) * self.dt_s, 6)
        lob_pose = interpolate_pose(stand, final_pose, 0.20)
        lob_pose = replace(
            lob_pose,
            pelvis_z_m=stand.pelvis_z_m - 0.06,
            leg_follow_pelvis=0.45,
            left_shoulder_abduction_rad=0.32,
            right_shoulder_abduction_rad=0.32,
        )
        self.emit_transition(
            lob_pose,
            0.22,
            action_label=fall_style,
            action_phase="loss_of_balance",
            fall_event_id=fall_event_id,
            fall_style=fall_style,
            intentional_contact=True,
        )

        # Phase 2: descent_rotation (gravity-like accelerating drop; never teleports)
        phase_ts["descent_rotation"] = round(len(self.frames) * self.dt_s, 6)
        eff_descent = max(0.30, descent_duration_s)
        if fall_style == "fall_slow_collapse":
            eff_descent = max(0.70, eff_descent)
        # Simulated downward contact speed estimate (explicitly tagged simulated, not physical measurement)
        delta_z = max(0.1, lob_pose.pelvis_z_m - final_pose.pelvis_z_m)
        sim_impact_speed = round(1.8 * delta_z / eff_descent, 4)

        self.emit_transition(
            final_pose,
            eff_descent,
            action_label=fall_style,
            action_phase="descent_rotation",
            fall_event_id=fall_event_id,
            fall_style=fall_style,
            intentional_contact=False,
            ease_power=1.15,
        )

        # Phase 3: impact_contact
        phase_ts["impact_contact"] = round(len(self.frames) * self.dt_s, 6)
        impact_time_s = phase_ts["impact_contact"]
        impact_pose = replace(final_pose, pelvis_z_m=max(0.105, final_pose.pelvis_z_m - 0.015))
        self.emit_transition(
            impact_pose,
            0.10,
            action_label=fall_style,
            action_phase="impact_contact",
            fall_event_id=fall_event_id,
            fall_style=fall_style,
            intentional_contact=False,
            impact_speed_mps=sim_impact_speed,
        )

        # Phase 4: settling (damped floor settling)
        phase_ts["settling"] = round(len(self.frames) * self.dt_s, 6)
        self.emit_transition(
            final_pose,
            0.45,
            action_label=fall_style,
            action_phase="settling",
            fall_event_id=fall_event_id,
            fall_style=fall_style,
            intentional_contact=False,
        )

        # Phase 5: optional_recovery (either gradual push-up recovery or floor dwell)
        phase_ts["optional_recovery"] = round(len(self.frames) * self.dt_s, 6)
        if recover:
            kneel_mid = PoseState(
                root_x_m=stand.root_x_m,
                root_y_m=stand.root_y_m,
                root_yaw_rad=fall_yaw_rad,
                pelvis_z_m=0.30 * self.height_m,
                torso_pitch_rad=0.55,
                left_hip_pitch_rad=0.60,
                left_knee_pitch_rad=1.55,
                right_hip_pitch_rad=0.80,
                right_knee_pitch_rad=1.45,
                left_shoulder_pitch_rad=0.95,
                right_shoulder_pitch_rad=0.95,
            )
            self.emit_transition(
                kneel_mid,
                0.85,
                action_label=fall_style,
                action_phase="optional_recovery",
                fall_event_id=fall_event_id,
                fall_style=fall_style,
                intentional_contact=False,
            )
            self.emit_transition(
                stand,
                0.85,
                action_label=fall_style,
                action_phase="optional_recovery",
                fall_event_id=fall_event_id,
                fall_style=fall_style,
                intentional_contact=True,
            )
        else:
            # Remain still on floor with subtle breathing
            n_still = int(round(1.6 * self.sample_rate_hz))
            for k in range(n_still):
                t = k * self.dt_s
                p = replace(final_pose, torso_pitch_rad=final_pose.torso_pitch_rad + 0.01 * math.sin(2.0 * math.pi * 0.25 * t))
                self.append_frame(
                    p,
                    action_label=fall_style,
                    action_phase="optional_recovery",
                    fall_event_id=fall_event_id,
                    fall_style=fall_style,
                    intentional_contact=False,
                )

        end_frame = len(self.frames) - 1
        seg = ActionSegment(
            action_label=fall_style,
            start_time_s=start_time,
            end_time_s=round(end_frame * self.dt_s, 6),
            start_frame=start_frame,
            end_frame=end_frame,
            is_fall=True,
            is_hard_negative=False,
            fall_event_id=fall_event_id,
            fall_style=fall_style,
            phase_timestamps_s=phase_ts,
        )
        self.events.append(seg)
        self.fall_events.append(
            {
                "fall_event_id": fall_event_id,
                "fall_style": fall_style,
                "start_time_s": start_time,
                "impact_time_s": impact_time_s,
                "end_time_s": seg.end_time_s,
                "recovered": recover,
                "simulated_impact_speed_mps": sim_impact_speed,
                "impact_status": "simulated_contact_quantity_not_physical_measurement",
                "phase_timestamps_s": phase_ts,
            }
        )
        return seg


def generate_motion_sequence(
    scene: Scene,
    motion_config: MotionConfig,
    *,
    seed: int | None = None,
    scenario_mode: str | None = None,
    fall_style: str | None = None,
    max_duration_scale: float = 1.0,
) -> MotionSequence:
    """Generate a complete deterministic 100 Hz motion sequence for the given scene and config."""
    mode = motion_config.scenario.default_mode if scenario_mode is None else scenario_mode
    if mode not in motion_config.scenario.modes:
        raise ValueError(
            f"Unknown scenario_mode {mode!r}; expected one of {sorted(motion_config.scenario.modes)}"
        )
    if fall_style is not None and fall_style not in FALL_STYLES:
        raise ValueError(f"Unknown fall_style {fall_style!r}; expected one of {list(FALL_STYLES)}")

    seed_mgr = SeedManager.from_motion_config(motion_config, override_seed=seed)
    rng_body = seed_mgr.rng("body_shape")
    rng_speed = seed_mgr.rng("speed")
    rng_path = seed_mgr.rng("path")
    rng_order = seed_mgr.rng("action_order")
    rng_fall = seed_mgr.rng("fall_type")

    # 1. Sample actor body dimensions
    height_m = float(rng_body.uniform(*motion_config.actor.height_m_range))
    body_scale = float(rng_body.uniform(*motion_config.actor.body_scale_range))
    body_radius_m = 0.16 * body_scale

    navigator = AStarNavigator(
        scene,
        resolution_m=motion_config.motion.navigation_grid_resolution_m,
        body_radius_m=body_radius_m,
        obstacle_clearance_m=motion_config.motion.obstacle_clearance_m,
    )

    # Pre-compute furniture interaction spots from the normalized Scene
    sit_spots: list[FurnitureInteractionSpot] = []
    lie_spots: list[FurnitureInteractionSpot] = []
    for obj in scene.objects:
        if obj.kind in ("chair", "sofa", "bed"):
            spot = navigator.find_furniture_interaction(obj, mode="sit")
            if spot is not None:
                sit_spots.append(spot)
        if obj.kind in ("bed", "sofa"):
            lspot = navigator.find_furniture_interaction(obj, mode="lie")
            if lspot is not None:
                lie_spots.append(lspot)

    # 2. Initial pose outside entrance doorway (or room center fallback)
    if scene.initial_actor is not None:
        start_xy: XY = (scene.initial_actor.start_position_m[0], scene.initial_actor.start_position_m[1])
        entry_xy: XY = (scene.initial_actor.entry_target_m[0], scene.initial_actor.entry_target_m[1])
        avatar_asset_id = scene.initial_actor.avatar_asset_id
    else:
        doors = scene.doors()
        if doors:
            dc = doors[0].center_m
            start_xy = (dc[0], -0.75)
            entry_xy = (dc[0], 0.55)
        else:
            start_xy = (0.5 * scene.width_x_m, 0.5 * scene.depth_y_m)
            entry_xy = start_xy
        avatar_asset_id = AVATAR_ASSET_ID

    init_yaw = _yaw_from_delta(entry_xy[0] - start_xy[0], entry_xy[1] - start_xy[1], fallback=0.0)
    init_pose = standing_pose(start_xy[0], start_xy[1], init_yaw, height_m)

    builder = _MotionBuilder(
        scene=scene,
        sample_rate_hz=motion_config.actor.motion_sample_rate_hz,
        actor_id=motion_config.actor.actor_id,
        avatar_asset_id=avatar_asset_id,
        scenario_mode=mode,
        height_m=height_m,
        body_scale=body_scale,
        initial_pose=init_pose,
    )

    # Emit initial frame at t = 0.0 s
    builder.append_frame(init_pose, action_label="enter")

    # 3. Entry through doorway
    if motion_config.scenario.entry_required and start_xy != entry_xy:
        entry_speed = float(rng_speed.uniform(*motion_config.motion.walking_speed_mps_range))
        builder.emit_walk_along_path([start_xy, entry_xy], entry_speed, action_label="enter")

    # 4. Decide whether this run includes a fall based on scenario_mode
    mode_cfg = motion_config.scenario.modes[mode]
    if mode_cfg.exactly_one_fall or mode_cfg.fall_probability >= 1.0:
        should_fall = True
    elif mode_cfg.fall_probability <= 0.0:
        should_fall = False
    else:
        should_fall = bool(rng_fall.random() < mode_cfg.fall_probability)

    chosen_fall_style = (
        fall_style
        if fall_style is not None
        else str(FALL_STYLES[int(rng_fall.integers(0, len(FALL_STYLES)))])
    )
    fall_side_sign = 1.0 if float(rng_fall.random()) < 0.5 else -1.0
    fall_recover = bool(rng_fall.random() < motion_config.motion.fall_recovery_probability)

    # 5. Build randomized non-fall action schedule from weighted_actions + hard_negatives
    min_act = motion_config.scenario.minimum_actions_per_run
    max_act = motion_config.scenario.maximum_actions_per_run
    total_actions = int(rng_order.integers(min_act, max_act + 1))
    remaining_actions = max(2, total_actions - 1)  # 1 already used by 'enter'

    non_fall_pool: list[tuple[str, float]] = [
        (name, w)
        for name, w in motion_config.weighted_actions.items()
        if name != "enter" and not name.startswith("fall_") and name != "stand" and w > 0.0
    ]
    if not motion_config.scenario.allow_sleep:
        non_fall_pool = [(n, w) for n, w in non_fall_pool if n != "sleep"]
    if not motion_config.scenario.allow_intentional_lying_down:
        non_fall_pool = [(n, w) for n, w in non_fall_pool if n != "lie_down_intentionally"]

    action_names = [n for n, _ in non_fall_pool]
    weights = np.array([w for _, w in non_fall_pool], dtype=np.float64)
    weights /= weights.sum()

    planned_actions: list[str] = list(
        rng_order.choice(action_names, size=remaining_actions, replace=True, p=weights)
    )

    # Ensure non-fall runs include at least one explicit hard-negative action distinct from falls
    if not should_fall and motion_config.hard_negative_actions:
        hn_choice = str(
            motion_config.hard_negative_actions[
                int(rng_order.integers(0, len(motion_config.hard_negative_actions)))
            ]
        )
        insert_idx = int(rng_order.integers(1, max(2, len(planned_actions))))
        planned_actions[min(insert_idx, len(planned_actions) - 1)] = hn_choice

    # Insert the single fall event near the latter half of the sequence if should_fall
    if should_fall:
        fall_idx = max(1, len(planned_actions) - 1)
        planned_actions = planned_actions[:fall_idx] + [chosen_fall_style]

    free_points = navigator.free_cells_xy(min_wall_margin_m=0.35)
    if not free_points:
        free_points = navigator.free_cells_xy()

    def _walk_to(target_xy: XY, slow: bool = False) -> bool:
        cur_xy = (builder.pose.root_x_m, builder.pose.root_y_m)
        if math.hypot(target_xy[0] - cur_xy[0], target_xy[1] - cur_xy[1]) < 0.12:
            return True
        path = navigator.plan_path(cur_xy, target_xy)
        if path is None:
            return False
        spd_range = (
            motion_config.motion.slow_walk_speed_mps_range
            if slow
            else motion_config.motion.walking_speed_mps_range
        )
        speed = float(rng_speed.uniform(*spd_range))
        builder.emit_walk_along_path(path, speed, action_label="walk")
        return True

    def _pick_free_target() -> XY:
        if not free_points:
            return (builder.pose.root_x_m, builder.pose.root_y_m)
        cur = (builder.pose.root_x_m, builder.pose.root_y_m)
        for _ in range(15):
            cand = free_points[int(rng_path.integers(0, len(free_points)))]
            if math.hypot(cand[0] - cur[0], cand[1] - cur[1]) >= 0.65:
                return cand
        return free_points[int(rng_path.integers(0, len(free_points)))]

    # 6. Execute each scheduled action
    fall_counter = 0
    for act in planned_actions:
        if act in FALL_STYLES:
            fall_counter += 1
            fall_id = f"fall_{seed_mgr.global_seed}_{fall_counter:02d}"
            safe_xy, safe_yaw = _find_safe_fall_spot(
                navigator,
                scene,
                (builder.pose.root_x_m, builder.pose.root_y_m),
                rng_path,
                rng_fall,
                height_m=height_m,
                body_scale=body_scale,
                fall_style=act,
                side_sign=fall_side_sign,
            )
            _walk_to(safe_xy)
            descent_s = float(rng_speed.uniform(*motion_config.motion.fall_duration_s_range))
            builder.emit_fall(
                fall_style=act,
                fall_event_id=fall_id,
                fall_yaw_rad=safe_yaw,
                descent_duration_s=descent_s,
                recover=fall_recover,
                side_sign=fall_side_sign,
            )
            continue

        if act == "walk":
            target = _pick_free_target()
            _walk_to(target, slow=bool(rng_speed.random() < 0.25))

        elif act == "turn":
            delta_yaw = float(rng_path.uniform(-math.pi, math.pi))
            if abs(delta_yaw) < 0.45:
                delta_yaw = math.copysign(0.8, delta_yaw or 1.0)
            target_yaw = math.atan2(
                math.sin(builder.pose.root_yaw_rad + delta_yaw),
                math.cos(builder.pose.root_yaw_rad + delta_yaw),
            )
            rate = float(rng_speed.uniform(*motion_config.motion.body_turn_rate_rad_per_s_range))
            builder.emit_turn(target_yaw, rate)

        elif act == "pause":
            dur = float(rng_speed.uniform(*motion_config.motion.pause_duration_s_range)) * max_duration_scale
            builder.emit_pause(max(0.3, dur))

        elif act in ("sit", "quick_sit", "sit_down_on_sofa_quickly"):
            pool = [s for s in sit_spots if s.object_kind == "sofa"] if act == "sit_down_on_sofa_quickly" else sit_spots
            if not pool:
                pool = sit_spots
            if pool:
                spot = pool[int(rng_path.integers(0, len(pool)))]
                if _walk_to(spot.approach_xy_m):
                    sit_dur = float(rng_speed.uniform(*motion_config.motion.sit_duration_s_range)) * max_duration_scale
                    builder.emit_sit_and_stand(
                        spot,
                        max(1.0, sit_dur),
                        sit_label=act,
                        is_hard_negative=(act != "sit"),
                        quick=(act != "sit"),
                    )
            else:
                builder.emit_pause(0.6)

        elif act in ("bend", "pick_up", "bend_to_pick_object"):
            target = _pick_free_target()
            _walk_to(target)
            builder.emit_bend_or_pickup(
                action_label=act,
                is_hard_negative=(act == "bend_to_pick_object"),
                hold_s=0.45,
            )

        elif act in ("kneel", "kneel_to_floor"):
            target = _pick_free_target()
            _walk_to(target)
            builder.emit_kneel(
                action_label=act,
                is_hard_negative=(act == "kneel_to_floor"),
                hold_s=0.80,
            )

        elif act in (
            "lie_down_intentionally",
            "intentional_lie_down",
            "sleep",
            "sleep_and_turn",
            "roll_over_while_lying",
        ):
            if lie_spots:
                spot = lie_spots[int(rng_path.integers(0, len(lie_spots)))]
                if _walk_to(spot.approach_xy_m):
                    is_sleep = act in ("sleep", "sleep_and_turn")
                    dur_range = (
                        motion_config.motion.sleep_duration_s_range
                        if is_sleep
                        else motion_config.motion.lie_down_duration_s_range
                    )
                    dur = float(rng_speed.uniform(*dur_range)) * max_duration_scale
                    builder.emit_lie_or_sleep(
                        spot,
                        max(1.0, dur),
                        action_label=act,
                        is_hard_negative=True,
                        with_roll=(act in ("roll_over_while_lying", "sleep_and_turn")),
                    )
            else:
                builder.emit_pause(0.8)

        else:
            builder.emit_pause(0.5)

    return MotionSequence(
        seed=seed_mgr.global_seed,
        scenario_mode=mode,
        sample_rate_hz=motion_config.actor.motion_sample_rate_hz,
        actor_id=motion_config.actor.actor_id,
        avatar_asset_id=avatar_asset_id,
        height_m=height_m,
        body_scale=body_scale,
        frames=builder.frames,
        events=builder.events,
        seed_map=seed_mgr.seed_map(),
        fall_events=builder.fall_events,
    )
