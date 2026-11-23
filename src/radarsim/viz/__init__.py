"""Visualization backend: WebSocket frame server, control validation, and SimulationSession."""

from radarsim.viz.protocol import (
    ControlCommand,
    ControlMessageError,
    KNOWN_IMPAIRMENTS,
    VALID_MODES,
    build_simulation_frame_message,
    parse_control_message,
    select_display_rf_paths,
)
from radarsim.viz.session import SimulationSession

__all__ = [
    "ControlCommand",
    "ControlMessageError",
    "KNOWN_IMPAIRMENTS",
    "SimulationSession",
    "VALID_MODES",
    "build_simulation_frame_message",
    "parse_control_message",
    "select_display_rf_paths",
]
