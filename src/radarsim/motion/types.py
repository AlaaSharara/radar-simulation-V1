"""Shared motion data structures following ``INTERFACES.md`` §4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]

# Exact 17 joint names mandated by INTERFACES.md §4 (order matches root-first rig tree)
JOINT_NAMES: tuple[str, ...] = (
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)

JOINT_PARENTS: dict[str, str | None] = {
    "pelvis": None,
    "spine": "pelvis",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    "left_shoulder": "chest",
    "left_elbow": "left_shoulder",
    "left_wrist": "left_elbow",
    "right_shoulder": "chest",
    "right_elbow": "right_shoulder",
    "right_wrist": "right_elbow",
    "left_hip": "pelvis",
    "left_knee": "left_hip",
    "left_ankle": "left_knee",
    "right_hip": "pelvis",
    "right_knee": "right_hip",
    "right_ankle": "right_knee",
}

FALL_STYLES: tuple[str, ...] = (
    "fall_forward",
    "fall_backward",
    "fall_sideways",
    "fall_slump",
    "fall_slow_collapse",
)

FALL_PHASES: tuple[str, ...] = (
    "loss_of_balance",
    "descent_rotation",
    "impact_contact",
    "settling",
    "optional_recovery",
)


@dataclass(frozen=True)
class BodyCapsule:
    """Collision and RF-scattering capsule proxy for a body segment (never drawn as avatar)."""

    segment_id: str
    a_m: Vec3
    b_m: Vec3
    radius_m: float
    material_id: str
    rcs_prior_dbsm: float = -15.0
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    rcs_provenance: str = "broad_assumption_prior_not_a_measured_human_rcs_dataset"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center_m(self) -> Vec3:
        return (
            0.5 * (self.a_m[0] + self.b_m[0]),
            0.5 * (self.a_m[1] + self.b_m[1]),
            0.5 * (self.a_m[2] + self.b_m[2]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "a_m": list(self.a_m),
            "b_m": list(self.b_m),
            "radius_m": self.radius_m,
            "material_id": self.material_id,
            "rcs_prior_dbsm": self.rcs_prior_dbsm,
            "velocity_mps": list(self.velocity_mps),
            "rcs_provenance": self.rcs_provenance,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContactEvent:
    """Floor or furniture contact state for a body part at a given frame."""

    contact_id: str
    surface_kind: str
    body_part: str
    contact_point_m: Vec3
    intentional: bool
    impact_speed_mps: float = 0.0
    impact_status: str = "simulated_contact_quantity_not_physical_measurement"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "surface_kind": self.surface_kind,
            "body_part": self.body_part,
            "contact_point_m": list(self.contact_point_m),
            "intentional": self.intentional,
            "impact_speed_mps": self.impact_speed_mps,
            "impact_status": self.impact_status,
        }


@dataclass(frozen=True)
class MotionFrame:
    """Single 100 Hz motion state adhering to ``INTERFACES.md`` §4."""

    frame_id: int
    sim_time_s: float
    actor_id: str
    action_label: str
    scenario_mode: str
    fall_event_id: str | None
    root_position_m: Vec3
    root_yaw_rad: float
    joints_m: dict[str, Vec3]
    body_capsules: list[BodyCapsule]
    contacts: list[ContactEvent]
    avatar_asset_id: str
    action_phase: str | None = None
    is_hard_negative: bool = False
    fall_style: str | None = None

    def to_ground_truth_dict(self) -> dict[str, Any]:
        """Serialize to the ``ground_truth`` block in ``INTERFACES.md`` §8."""
        return {
            "actor_id": self.actor_id,
            "action_label": self.action_label,
            "action_phase": self.action_phase,
            "scenario_mode": self.scenario_mode,
            "fall_event_id": self.fall_event_id,
            "fall_style": self.fall_style,
            "is_hard_negative": self.is_hard_negative,
            "root_position_m": list(self.root_position_m),
            "root_yaw_rad": self.root_yaw_rad,
            "joints_m": {k: list(v) for k, v in self.joints_m.items()},
            "contacts": [c.to_dict() for c in self.contacts],
            "avatar_asset_id": self.avatar_asset_id,
        }


@dataclass(frozen=True)
class ActionSegment:
    """Metadata for one scheduled action in a motion sequence."""

    action_label: str
    start_time_s: float
    end_time_s: float
    start_frame: int
    end_frame: int
    is_fall: bool = False
    is_hard_negative: bool = False
    fall_event_id: str | None = None
    fall_style: str | None = None
    target_object_id: str | None = None
    phase_timestamps_s: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_label": self.action_label,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "is_fall": self.is_fall,
            "is_hard_negative": self.is_hard_negative,
            "fall_event_id": self.fall_event_id,
            "fall_style": self.fall_style,
            "target_object_id": self.target_object_id,
            "phase_timestamps_s": dict(self.phase_timestamps_s),
        }


@dataclass(frozen=True)
class MotionSequence:
    """Complete deterministic 100 Hz motion sequence for one scenario run."""

    seed: int
    scenario_mode: str
    sample_rate_hz: float
    actor_id: str
    avatar_asset_id: str
    height_m: float
    body_scale: float
    frames: list[MotionFrame]
    events: list[ActionSegment]
    seed_map: dict[str, int]
    fall_events: list[dict[str, Any]]

    @property
    def duration_s(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].sim_time_s

    def frame_at_time(self, sim_time_s: float) -> MotionFrame:
        if not self.frames:
            raise ValueError("MotionSequence contains no frames")
        idx = int(round(sim_time_s * self.sample_rate_hz))
        idx = max(0, min(len(self.frames) - 1, idx))
        return self.frames[idx]

    def sample_at_rate(self, target_fps_hz: float) -> list[MotionFrame]:
        """Subsample frames deterministically to a radar/dashboard frame rate (e.g., 20 Hz)."""
        if target_fps_hz <= 0.0:
            raise ValueError(f"target_fps_hz must be > 0, got {target_fps_hz}")
        if not self.frames:
            return []
        dt = 1.0 / target_fps_hz
        total_t = self.duration_s
        sampled: list[MotionFrame] = []
        k = 0
        while k * dt <= total_t + 1e-9:
            sampled.append(self.frame_at_time(k * dt))
            k += 1
        return sampled
