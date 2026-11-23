"""Room-scale geometric-optics and stochastic-diffuse RF propagation engine."""

from radarsim.rf.antenna import antenna_gain_linear, tx_rx_world_positions
from radarsim.rf.em import (
    SurfaceEMResponse,
    evaluate_surface_em,
    fresnel_single_interface,
    multilayer_reflection_transmission,
    rayleigh_roughness_factor,
)
from radarsim.rf.engine import RFPropagationEngine
from radarsim.rf.raytrace import (
    FirstOrderReflection,
    SecondOrderReflection,
    SurfaceHit,
    SurfaceRayAccelerator,
    find_first_order_reflection,
    find_second_order_reflection,
    find_segment_surface_hits,
    is_segment_occluded,
    is_segment_occluded_by_capsules,
    mirror_point_across_plane,
    ray_intersect_surface,
)
from radarsim.rf.types import (
    PathSet,
    PointTarget,
    RFPath,
    RFPropagationConfig,
    VALID_PATH_TYPES,
)

__all__ = [
    "FirstOrderReflection",
    "PathSet",
    "PointTarget",
    "RFPath",
    "RFPropagationConfig",
    "RFPropagationEngine",
    "SecondOrderReflection",
    "SurfaceEMResponse",
    "SurfaceHit",
    "SurfaceRayAccelerator",
    "VALID_PATH_TYPES",
    "antenna_gain_linear",
    "evaluate_surface_em",
    "find_first_order_reflection",
    "find_second_order_reflection",
    "find_segment_surface_hits",
    "fresnel_single_interface",
    "is_segment_occluded",
    "is_segment_occluded_by_capsules",
    "mirror_point_across_plane",
    "multilayer_reflection_transmission",
    "ray_intersect_surface",
    "rayleigh_roughness_factor",
    "tx_rx_world_positions",
]
