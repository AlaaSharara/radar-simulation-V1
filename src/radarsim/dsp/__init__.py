"""Radar-only DSP chain: Range/Doppler FFT, OS-CFAR, angle estimation, tracking, and fall baseline."""

from radarsim.dsp.angle import ApertureLimitations, VirtualArrayAngleEstimator
from radarsim.dsp.cfar import CFARPeak, OSCFARConfig, run_os_cfar_2d
from radarsim.dsp.fall import FallRuleConfig, RuleBasedFallDetector, StationaryPhaseBranch
from radarsim.dsp.processor import DSPConfig, DSPProcessor
from radarsim.dsp.spectral import SpectralCubeResult, WindowStats, compute_range_doppler_cubes, make_window
from radarsim.dsp.tracking import (
    DetectionCluster,
    RadarTracker,
    build_radar_occupancy_voxels,
    cluster_detections,
)
from radarsim.dsp.types import (
    DSPFrame,
    FallAlert,
    RadarDetection,
    RadarOccupancyVoxels,
    RadarTrack,
    RangeAngleMap,
    RangeDopplerMap,
)

__all__ = [
    "ApertureLimitations",
    "CFARPeak",
    "DSPConfig",
    "DSPFrame",
    "DSPProcessor",
    "DetectionCluster",
    "FallAlert",
    "FallRuleConfig",
    "OSCFARConfig",
    "RadarDetection",
    "RadarOccupancyVoxels",
    "RadarTrack",
    "RadarTracker",
    "RangeAngleMap",
    "RangeDopplerMap",
    "RuleBasedFallDetector",
    "SpectralCubeResult",
    "StationaryPhaseBranch",
    "VirtualArrayAngleEstimator",
    "WindowStats",
    "build_radar_occupancy_voxels",
    "cluster_detections",
    "compute_range_doppler_cubes",
    "make_window",
    "run_os_cfar_2d",
]
