"""Furnished room geometry, material-aware surfaces, collision boxes, and layout generation."""

from radarsim.scene.generator import (
    GeneratedLayoutResult,
    LayoutGenerationBounds,
    generate_layout_batch,
    generate_random_layout,
)
from radarsim.scene.geometry import (
    AABB,
    OrientedBox,
    RadarCoordinateFrame,
    SurfaceLayer,
    SurfacePatch,
    azimuth_elevation_from_local,
    boresight_from_yaw_tilt,
    local_from_range_azimuth_elevation,
    radial_velocity_mps,
)
from radarsim.scene.materials import (
    EPSILON_0_F_PER_M,
    MaterialProperties,
    complex_permittivity,
    conductivity_s_per_m,
    lookup_material_properties,
)
from radarsim.scene.model import (
    OccupancyGrid2D,
    Scene,
    SceneObject,
    SceneOpening,
    build_scene,
    load_scene,
)
from radarsim.scene.validation import (
    GeometryIssue,
    SceneValidationReport,
    doorway_clearance_box,
    validate_scene_geometry,
)

__all__ = [
    "AABB",
    "EPSILON_0_F_PER_M",
    "GeneratedLayoutResult",
    "GeometryIssue",
    "LayoutGenerationBounds",
    "MaterialProperties",
    "OccupancyGrid2D",
    "OrientedBox",
    "RadarCoordinateFrame",
    "Scene",
    "SceneObject",
    "SceneOpening",
    "SceneValidationReport",
    "SurfaceLayer",
    "SurfacePatch",
    "azimuth_elevation_from_local",
    "boresight_from_yaw_tilt",
    "build_scene",
    "complex_permittivity",
    "conductivity_s_per_m",
    "doorway_clearance_box",
    "generate_layout_batch",
    "generate_random_layout",
    "load_scene",
    "local_from_range_azimuth_elevation",
    "lookup_material_properties",
    "radial_velocity_mps",
    "validate_scene_geometry",
]
