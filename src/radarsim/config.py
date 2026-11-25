"""Strict configuration loading, validation, and independent derived radar metrics.

All physical quantities are stored in SI units (metres, seconds, hertz, radians,
watts, volts, ohms, kelvin) except where configuration fields explicitly state
another unit (for example, dBm, dBi, dB, degrees, wavelengths, or ppm).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import yaml


DERIVED_METRIC_KEYS: tuple[str, ...] = (
    "effective_bandwidth_hz",
    "range_resolution_m",
    "max_unambiguous_range_m",
    "wavelength_m",
    "same_tx_chirp_interval_s",
    "max_unambiguous_radial_velocity_mps",
    "velocity_resolution_mps",
)

VALID_WINDOWS: frozenset[str] = frozenset(
    {"hann", "hamming", "blackman", "rectangular", "none"}
)
VALID_OPENING_KINDS: frozenset[str] = frozenset({"door", "window"})
VALID_WALLS: frozenset[str] = frozenset({"x_min", "x_max", "y_min", "y_max"})
VALID_POLARIZATIONS: frozenset[str] = frozenset({"TE", "TM"})
VALID_COORDINATE_UNITS: frozenset[str] = frozenset({"wavelengths", "m", "meters", "metres"})


class ConfigValidationError(ValueError):
    """Raised when a configuration file is missing required fields or has invalid values."""

    def __init__(
        self,
        message: str,
        *,
        source: str | None = None,
        field_path: str | None = None,
    ) -> None:
        self.source = source
        self.field_path = field_path
        prefix_parts: list[str] = []
        if source:
            prefix_parts.append(f"[{source}]")
        if field_path:
            prefix_parts.append(f"Field '{field_path}':")
        full_message = " ".join([*prefix_parts, message]) if prefix_parts else message
        super().__init__(full_message)


def _fail(source: str, field_path: str, message: str) -> ConfigValidationError:
    return ConfigValidationError(message, source=source, field_path=field_path)


def _join_path(prefix: str, key: str) -> str:
    return f"{prefix}.{key}" if prefix else key


def _require_mapping(
    obj: Any, source: str, field_path: str
) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise _fail(
            source,
            field_path,
            f"expected a mapping/object, got {type(obj).__name__}",
        )
    return obj


def _require_key(
    mapping: Mapping[str, Any], key: str, source: str, prefix: str = ""
) -> Any:
    path = _join_path(prefix, key)
    if key not in mapping:
        raise ConfigValidationError(
            f"Missing required field '{path}'",
            source=source,
        )
    return mapping[key]


def _require_str(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    non_empty: bool = True,
) -> str:
    path = _join_path(prefix, key)
    value = _require_key(mapping, key, source, prefix)
    if not isinstance(value, str):
        raise _fail(source, path, f"expected a string, got {type(value).__name__}")
    if non_empty and not value.strip():
        raise _fail(source, path, "must be a non-empty string")
    return value


def _optional_str(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
) -> str | None:
    if key not in mapping or mapping[key] is None:
        return None
    path = _join_path(prefix, key)
    value = mapping[key]
    if not isinstance(value, str):
        raise _fail(source, path, f"expected a string or null, got {type(value).__name__}")
    return value


def _require_bool(
    mapping: Mapping[str, Any], key: str, source: str, prefix: str = ""
) -> bool:
    path = _join_path(prefix, key)
    value = _require_key(mapping, key, source, prefix)
    if not isinstance(value, bool):
        raise _fail(source, path, f"expected a boolean, got {type(value).__name__}")
    return value


def _optional_bool(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    default: bool | None = None,
) -> bool | None:
    if key not in mapping or mapping[key] is None:
        return default
    path = _join_path(prefix, key)
    value = mapping[key]
    if not isinstance(value, bool):
        raise _fail(source, path, f"expected a boolean, got {type(value).__name__}")
    return value


def _as_int(
    value: Any,
    source: str,
    path: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(source, path, f"expected an integer, got {value!r} ({type(value).__name__})")
    if min_value is not None and value < min_value:
        raise _fail(source, path, f"must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise _fail(source, path, f"must be <= {max_value}, got {value}")
    return value


def _require_int(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    path = _join_path(prefix, key)
    value = _require_key(mapping, key, source, prefix)
    return _as_int(value, source, path, min_value=min_value, max_value=max_value)


def _as_float(
    value: Any,
    source: str,
    path: str,
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(source, path, f"expected a finite number, got {value!r} ({type(value).__name__})")
    num = float(value)
    if not math.isfinite(num):
        raise _fail(source, path, f"expected a finite number, got {value!r}")
    if gt is not None and not (num > gt):
        raise _fail(source, path, f"must be > {gt}, got {num}")
    if ge is not None and not (num >= ge):
        raise _fail(source, path, f"must be >= {ge}, got {num}")
    if lt is not None and not (num < lt):
        raise _fail(source, path, f"must be < {lt}, got {num}")
    if le is not None and not (num <= le):
        raise _fail(source, path, f"must be <= {le}, got {num}")
    return num


def _require_float(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> float:
    path = _join_path(prefix, key)
    value = _require_key(mapping, key, source, prefix)
    return _as_float(value, source, path, gt=gt, ge=ge, lt=lt, le=le)


def _optional_float(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    path = _join_path(prefix, key)
    return _as_float(mapping[key], source, path, gt=gt, ge=ge, lt=lt, le=le)


def _require_list(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    non_empty: bool = False,
    exact_length: int | None = None,
) -> list[Any]:
    path = _join_path(prefix, key)
    value = _require_key(mapping, key, source, prefix)
    if not isinstance(value, list):
        raise _fail(source, path, f"expected a list, got {type(value).__name__}")
    if non_empty and len(value) == 0:
        raise _fail(source, path, "must be a non-empty list")
    if exact_length is not None and len(value) != exact_length:
        raise _fail(source, path, f"expected list of length {exact_length}, got {len(value)}")
    return value


def _require_vec3(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    positive: bool = False,
) -> tuple[float, float, float]:
    path = _join_path(prefix, key)
    raw = _require_list(mapping, key, source, prefix, exact_length=3)
    gt = 0.0 if positive else None
    return (
        _as_float(raw[0], source, f"{path}[0]", gt=gt),
        _as_float(raw[1], source, f"{path}[1]", gt=gt),
        _as_float(raw[2], source, f"{path}[2]", gt=gt),
    )


def _require_range_pair(
    mapping: Mapping[str, Any],
    key: str,
    source: str,
    prefix: str = "",
    *,
    gt: float | None = None,
    ge: float | None = None,
) -> tuple[float, float]:
    path = _join_path(prefix, key)
    raw = _require_list(mapping, key, source, prefix, exact_length=2)
    lo = _as_float(raw[0], source, f"{path}[0]", gt=gt, ge=ge)
    hi = _as_float(raw[1], source, f"{path}[1]", gt=gt, ge=ge)
    if lo > hi:
        raise _fail(source, path, f"range lower bound ({lo}) must be <= upper bound ({hi})")
    return (lo, hi)


def _optional_str_tuple(
    mapping: Mapping[str, Any], key: str, source: str, prefix: str = ""
) -> tuple[str, ...]:
    if key not in mapping or mapping[key] is None:
        return ()
    path = _join_path(prefix, key)
    raw = mapping[key]
    if not isinstance(raw, list):
        raise _fail(source, path, f"expected a list of strings, got {type(raw).__name__}")
    out: list[str] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise _fail(source, f"{path}[{idx}]", "expected a non-empty string")
        out.append(item)
    return tuple(out)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigValidationError(f"Configuration file not found: {path}", source=path.name)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigValidationError(f"Invalid YAML syntax: {exc}", source=path.name) from exc
    return _require_mapping(raw, path.name, "<root>")


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigValidationError(f"Configuration file not found: {path}", source=path.name)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigValidationError(f"Invalid JSON syntax: {exc}", source=path.name) from exc
    return _require_mapping(raw, path.name, "<root>")


# ---------------------------------------------------------------------------
# Vendor-neutral RadarProfile and independent derived radar metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProfileMetadata:
    id: str
    name: str
    status: str
    hardware_reference: str | None = None
    source_urls: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class AntennaPatternConfig:
    model: str
    azimuth_fov_3db_deg: float
    elevation_fov_3db_deg: float
    cosine_power_exponent: float
    status: str | None = None


@dataclass(frozen=True)
class PhaseNoiseConfig:
    enabled: bool
    reference_dbc_per_hz_at_1mhz: float
    model: str
    linewidth_hz: float
    provenance: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class ReceiverConfig:
    noise_temperature_k: float
    noise_equivalent_bandwidth_hz: float
    max_if_frequency_hz: float
    if_filter_order: int
    if_filter_type: str
    receiver_gain_db: float
    adc_full_scale_vpp: float
    adc_input_impedance_ohm: float
    adc_bits: int
    adc_clip_enabled: bool
    adc_quantization_enabled: bool
    sample_representation: str
    receiver_gain_status: str | None = None
    adc_bits_status: str | None = None


@dataclass(frozen=True)
class RFConfig:
    speed_of_light_m_per_s: float
    carrier_frequency_hz: float
    operating_band_hz: tuple[float, float]
    tx_count: int
    rx_count: int
    tx_schedule: tuple[int, ...]
    tdm_mode: bool
    tx_power_dbm_per_active_tx: float
    antenna_gain_dbi_peak: float
    antenna_pattern: AntennaPatternConfig
    receiver_noise_figure_db: float
    phase_noise: PhaseNoiseConfig
    receiver: ReceiverConfig
    tx_power_provenance: str | None = None
    antenna_gain_provenance: str | None = None
    receiver_noise_figure_provenance: str | None = None


@dataclass(frozen=True)
class WindowConfig:
    range: str
    doppler: str
    angle: str


@dataclass(frozen=True)
class WaveformConfig:
    type: str
    start_frequency_hz: float
    slope_hz_per_s: float
    ramp_time_s: float
    adc_start_time_s: float
    adc_sample_rate_hz: float
    adc_samples_per_chirp: int
    adc_sampling: str
    idle_time_s: float
    chirp_period_s: float
    chirps_per_tx_per_frame: int
    frame_period_s: float
    frame_rate_hz: float
    nominal_sweep_bandwidth_hz: float
    range_fft_size: int
    doppler_fft_size: int
    angle_fft_azimuth_size: int
    angle_fft_elevation_size: int
    window: WindowConfig


@dataclass(frozen=True)
class AntennaArrayConfig:
    layout: str
    coordinate_units: str
    rx_positions_lambda: tuple[tuple[float, float, float], ...]
    tx_positions_lambda: tuple[tuple[float, float, float], ...]
    virtual_position_rule: str
    channel_gain_mismatch_db_rms: float
    channel_phase_mismatch_deg_rms: float
    status: str | None = None
    channel_mismatch_status: str | None = None


@dataclass(frozen=True)
class ImpairmentsConfig:
    phase_noise_enabled: bool
    phase_noise_strength_scale: float
    chirp_slope_error_enabled: bool
    chirp_slope_error_sigma_ppm: float
    chirp_start_frequency_error_enabled: bool
    chirp_start_frequency_error_sigma_hz: float
    tx_rx_channel_mismatch_enabled: bool
    iq_imbalance_enabled: bool
    iq_imbalance_amplitude_mismatch_db_sigma: float
    iq_imbalance_phase_mismatch_deg_sigma: float
    adc_quantization_enabled: bool
    adc_clipping_enabled: bool
    tx_leakage_enabled: bool
    tx_leakage_level_db_relative_to_full_scale: float
    timing_jitter_enabled: bool
    timing_jitter_sigma_s: float
    thermal_noise_enabled: bool
    provenance: str | None = None

    def as_toggle_dict(self) -> dict[str, bool]:
        return {
            "phase_noise": self.phase_noise_enabled,
            "chirp_slope_error": self.chirp_slope_error_enabled,
            "chirp_start_frequency_error": self.chirp_start_frequency_error_enabled,
            "tx_rx_channel_mismatch": self.tx_rx_channel_mismatch_enabled,
            "iq_imbalance": self.iq_imbalance_enabled,
            "adc_quantization": self.adc_quantization_enabled,
            "adc_clipping": self.adc_clipping_enabled,
            "tx_leakage": self.tx_leakage_enabled,
            "timing_jitter": self.timing_jitter_enabled,
            "thermal_noise": self.thermal_noise_enabled,
        }

    def enabled_names(self) -> tuple[str, ...]:
        return tuple(name for name, enabled in self.as_toggle_dict().items() if enabled)


@dataclass(frozen=True)
class RCSPriorConfig:
    minimum: float
    nominal: float
    maximum: float


@dataclass(frozen=True)
class HumanScatteringConfig:
    model: str
    rcs_prior_dbsm: RCSPriorConfig
    aspect_dependence_enabled: bool
    clothing_scattering_scale: float
    deterministic_micro_scatterers_per_body_segment: int
    random_scatterers_per_body_segment: int
    rcs_status: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class DerivedRadarMetrics:
    """Radar performance metrics derived purely from RF and waveform parameters."""

    effective_bandwidth_hz: float
    range_resolution_m: float
    max_unambiguous_range_m: float
    wavelength_m: float
    same_tx_chirp_interval_s: float
    max_unambiguous_radial_velocity_mps: float
    velocity_resolution_mps: float

    def as_dict(self) -> dict[str, float]:
        return {
            "effective_bandwidth_hz": self.effective_bandwidth_hz,
            "range_resolution_m": self.range_resolution_m,
            "max_unambiguous_range_m": self.max_unambiguous_range_m,
            "wavelength_m": self.wavelength_m,
            "same_tx_chirp_interval_s": self.same_tx_chirp_interval_s,
            "max_unambiguous_radial_velocity_mps": self.max_unambiguous_radial_velocity_mps,
            "velocity_resolution_mps": self.velocity_resolution_mps,
        }


@dataclass(frozen=True)
class ExpectedDerivedMetrics:
    """Configured expected derived metrics and absolute SI tolerances."""

    values: dict[str, float]
    tolerances: dict[str, float]


@dataclass(frozen=True)
class MetricComparison:
    name: str
    calculated: float
    expected: float
    abs_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class RadarProfile:
    """Vendor-neutral radar sensor profile constructed strictly from configuration.

    Intentionally excludes ``expected_derived`` so downstream calculations can
    never read expected values as calculation inputs.
    """

    schema_version: str
    profile: ProfileMetadata
    rf: RFConfig
    waveform: WaveformConfig
    antenna_array: AntennaArrayConfig
    impairments: ImpairmentsConfig
    human_scattering: HumanScatteringConfig

    @property
    def profile_id(self) -> str:
        return self.profile.id

    @property
    def derived(self) -> DerivedRadarMetrics:
        return compute_derived_metrics(self)

    @property
    def virtual_channel_count(self) -> int:
        return self.rf.tx_count * self.rf.rx_count

    @property
    def adc_cube_shape(self) -> tuple[int, int, int, int]:
        """Canonical in-memory ADC cube shape: [loops, tx, rx, adc_samples]."""
        return (
            self.waveform.chirps_per_tx_per_frame,
            self.rf.tx_count,
            self.rf.rx_count,
            self.waveform.adc_samples_per_chirp,
        )

    def virtual_positions_lambda(self) -> tuple[tuple[float, float, float], ...]:
        """Compute virtual array phase-centre positions in configured coordinate units."""
        if self.antenna_array.virtual_position_rule != "tx_position_plus_rx_position":
            raise ConfigValidationError(
                f"Unsupported virtual_position_rule: {self.antenna_array.virtual_position_rule!r}",
                field_path="antenna_array.virtual_position_rule",
            )
        positions: list[tuple[float, float, float]] = []
        for tx_pos in self.antenna_array.tx_positions_lambda:
            for rx_pos in self.antenna_array.rx_positions_lambda:
                positions.append(
                    (tx_pos[0] + rx_pos[0], tx_pos[1] + rx_pos[1], tx_pos[2] + rx_pos[2])
                )
        return tuple(positions)

    def virtual_positions_m(self) -> tuple[tuple[float, float, float], ...]:
        """Compute virtual array phase-centre positions in SI metres."""
        raw_positions = self.virtual_positions_lambda()
        if self.antenna_array.coordinate_units == "wavelengths":
            wavelength = compute_derived_metrics(self).wavelength_m
            return tuple(
                (p[0] * wavelength, p[1] * wavelength, p[2] * wavelength) for p in raw_positions
            )
        return raw_positions

    def with_impairment_toggle(self, name: str, enabled: bool) -> RadarProfile:
        """Return a new RadarProfile with one named impairment toggled."""
        field_map = {
            "phase_noise": "phase_noise_enabled",
            "chirp_slope_error": "chirp_slope_error_enabled",
            "chirp_start_frequency_error": "chirp_start_frequency_error_enabled",
            "tx_rx_channel_mismatch": "tx_rx_channel_mismatch_enabled",
            "iq_imbalance": "iq_imbalance_enabled",
            "adc_quantization": "adc_quantization_enabled",
            "adc_clipping": "adc_clipping_enabled",
            "tx_leakage": "tx_leakage_enabled",
            "timing_jitter": "timing_jitter_enabled",
            "thermal_noise": "thermal_noise_enabled",
        }
        if name not in field_map:
            raise ConfigValidationError(
                f"Unknown impairment {name!r}; expected one of {sorted(field_map)}",
                field_path="impairments",
            )
        if not isinstance(enabled, bool):
            raise ConfigValidationError(
                f"Impairment toggle for {name!r} must be a boolean",
                field_path=f"impairments.{name}.enabled",
            )
        updated_impairments = replace(self.impairments, **{field_map[name]: enabled})
        return replace(self, impairments=updated_impairments)


def compute_derived_metrics(profile: RadarProfile) -> DerivedRadarMetrics:
    """Independently compute radar derived metrics from RF and waveform config only."""
    rf = profile.rf
    wf = profile.waveform

    c = rf.speed_of_light_m_per_s
    fc = rf.carrier_frequency_hz
    slope = wf.slope_hz_per_s
    fs = wf.adc_sample_rate_hz
    n_adc = wf.adc_samples_per_chirp
    max_if = rf.receiver.max_if_frequency_hz
    chirp_period = wf.chirp_period_s
    n_doppler = wf.chirps_per_tx_per_frame
    tx_cycle_length = len(rf.tx_schedule) if rf.tdm_mode else 1

    active_adc_time_s = n_adc / fs
    effective_bandwidth_hz = slope * active_adc_time_s
    wavelength_m = c / fc
    range_resolution_m = c / (2.0 * effective_bandwidth_hz)
    max_unambiguous_range_m = c * max_if / (2.0 * slope)
    same_tx_chirp_interval_s = chirp_period * tx_cycle_length
    max_unambiguous_radial_velocity_mps = wavelength_m / (4.0 * same_tx_chirp_interval_s)
    velocity_resolution_mps = wavelength_m / (2.0 * n_doppler * same_tx_chirp_interval_s)

    return DerivedRadarMetrics(
        effective_bandwidth_hz=effective_bandwidth_hz,
        range_resolution_m=range_resolution_m,
        max_unambiguous_range_m=max_unambiguous_range_m,
        wavelength_m=wavelength_m,
        same_tx_chirp_interval_s=same_tx_chirp_interval_s,
        max_unambiguous_radial_velocity_mps=max_unambiguous_radial_velocity_mps,
        velocity_resolution_mps=velocity_resolution_mps,
    )


def compare_derived_metrics(
    actual: DerivedRadarMetrics, expected: ExpectedDerivedMetrics
) -> list[MetricComparison]:
    """Compare independently calculated metrics against configured expected values."""
    actual_dict = actual.as_dict()
    comparisons: list[MetricComparison] = []
    for key in DERIVED_METRIC_KEYS:
        calc_val = actual_dict[key]
        exp_val = expected.values[key]
        tol = expected.tolerances[key]
        abs_error = abs(calc_val - exp_val)
        comparisons.append(
            MetricComparison(
                name=key,
                calculated=calc_val,
                expected=exp_val,
                abs_error=abs_error,
                tolerance=tol,
                passed=abs_error <= tol,
            )
        )
    return comparisons


def validate_derived_metrics(
    actual: DerivedRadarMetrics,
    expected: ExpectedDerivedMetrics,
    *,
    source: str = "radar_config.yaml",
) -> list[MetricComparison]:
    """Validate that all derived metrics match expected values within tolerances."""
    comparisons = compare_derived_metrics(actual, expected)
    failures = [item for item in comparisons if not item.passed]
    if failures:
        details = "; ".join(
            f"{item.name}: calculated={item.calculated:.12g}, expected={item.expected:.12g}, "
            f"|error|={item.abs_error:.3g} > tol={item.tolerance:.3g}"
            for item in failures
        )
        raise ConfigValidationError(
            f"Derived radar metric tolerance check failed: {details}",
            source=source,
            field_path="expected_derived",
        )
    return comparisons


def parse_expected_derived(
    raw: Mapping[str, Any], *, source: str = "radar_config.yaml"
) -> ExpectedDerivedMetrics:
    expected_map = _require_mapping(
        _require_key(raw, "expected_derived", source), source, "expected_derived"
    )
    tol_map = _require_mapping(
        _require_key(expected_map, "tolerances", source, "expected_derived"),
        source,
        "expected_derived.tolerances",
    )
    values: dict[str, float] = {}
    tolerances: dict[str, float] = {}
    for key in DERIVED_METRIC_KEYS:
        values[key] = _require_float(
            expected_map, key, source, "expected_derived", gt=0.0
        )
        tolerances[key] = _require_float(
            tol_map, key, source, "expected_derived.tolerances", ge=0.0
        )
    return ExpectedDerivedMetrics(values=values, tolerances=tolerances)


def parse_radar_profile(
    raw: Mapping[str, Any],
    *,
    source: str = "radar_config.yaml",
    require_expected_derived: bool = False,
    validate_expected: bool = True,
) -> tuple[RadarProfile, ExpectedDerivedMetrics | None]:
    """Validate a raw radar configuration mapping and build a vendor-neutral RadarProfile."""
    root = _require_mapping(raw, source, "<root>")
    schema_version = _require_str(root, "schema_version", source)

    # profile section
    prof_map = _require_mapping(_require_key(root, "profile", source), source, "profile")
    profile_meta = ProfileMetadata(
        id=_require_str(prof_map, "id", source, "profile"),
        name=_require_str(prof_map, "name", source, "profile"),
        status=_require_str(prof_map, "status", source, "profile"),
        hardware_reference=_optional_str(prof_map, "hardware_reference", source, "profile"),
        source_urls=_optional_str_tuple(prof_map, "source_urls", source, "profile"),
        notes=_optional_str_tuple(prof_map, "notes", source, "profile"),
    )

    # rf section
    rf_map = _require_mapping(_require_key(root, "rf", source), source, "rf")
    c = _require_float(rf_map, "speed_of_light_m_per_s", source, "rf", gt=0.0)
    fc = _require_float(rf_map, "carrier_frequency_hz", source, "rf", gt=0.0)
    band = _require_range_pair(rf_map, "operating_band_hz", source, "rf", gt=0.0)
    if not (band[0] <= fc <= band[1]):
        raise _fail(
            source,
            "rf.carrier_frequency_hz",
            f"carrier_frequency_hz ({fc}) must lie within operating_band_hz [{band[0]}, {band[1]}]",
        )
    tx_count = _require_int(rf_map, "tx_count", source, "rf", min_value=1)
    rx_count = _require_int(rf_map, "rx_count", source, "rf", min_value=1)
    tdm_mode = _require_bool(rf_map, "tdm_mode", source, "rf")
    raw_schedule = _require_list(rf_map, "tx_schedule", source, "rf", non_empty=True)
    tx_schedule = tuple(
        _as_int(item, source, f"rf.tx_schedule[{idx}]", min_value=0, max_value=tx_count - 1)
        for idx, item in enumerate(raw_schedule)
    )
    if tdm_mode:
        if len(tx_schedule) != tx_count or set(tx_schedule) != set(range(tx_count)):
            raise _fail(
                source,
                "rf.tx_schedule",
                f"when tdm_mode is true, tx_schedule must schedule each TX 0..{tx_count - 1} once, got {list(tx_schedule)}",
            )

    ap_map = _require_mapping(
        _require_key(rf_map, "antenna_pattern", source, "rf"), source, "rf.antenna_pattern"
    )
    antenna_pattern = AntennaPatternConfig(
        model=_require_str(ap_map, "model", source, "rf.antenna_pattern"),
        azimuth_fov_3db_deg=_require_float(
            ap_map, "azimuth_fov_3db_deg", source, "rf.antenna_pattern", gt=0.0, le=360.0
        ),
        elevation_fov_3db_deg=_require_float(
            ap_map, "elevation_fov_3db_deg", source, "rf.antenna_pattern", gt=0.0, le=360.0
        ),
        cosine_power_exponent=_require_float(
            ap_map, "cosine_power_exponent", source, "rf.antenna_pattern", gt=0.0
        ),
        status=_optional_str(ap_map, "status", source, "rf.antenna_pattern"),
    )

    pn_map = _require_mapping(
        _require_key(rf_map, "phase_noise", source, "rf"), source, "rf.phase_noise"
    )
    phase_noise = PhaseNoiseConfig(
        enabled=_require_bool(pn_map, "enabled", source, "rf.phase_noise"),
        reference_dbc_per_hz_at_1mhz=_require_float(
            pn_map, "reference_dbc_per_hz_at_1mhz", source, "rf.phase_noise", le=0.0
        ),
        model=_require_str(pn_map, "model", source, "rf.phase_noise"),
        linewidth_hz=_require_float(pn_map, "linewidth_hz", source, "rf.phase_noise", gt=0.0),
        provenance=_optional_str(pn_map, "provenance", source, "rf.phase_noise"),
        status=_optional_str(pn_map, "status", source, "rf.phase_noise"),
    )

    rx_map = _require_mapping(
        _require_key(rf_map, "receiver", source, "rf"), source, "rf.receiver"
    )
    receiver = ReceiverConfig(
        noise_temperature_k=_require_float(
            rx_map, "noise_temperature_k", source, "rf.receiver", gt=0.0
        ),
        noise_equivalent_bandwidth_hz=_require_float(
            rx_map, "noise_equivalent_bandwidth_hz", source, "rf.receiver", gt=0.0
        ),
        max_if_frequency_hz=_require_float(
            rx_map, "max_if_frequency_hz", source, "rf.receiver", gt=0.0
        ),
        if_filter_order=_require_int(
            rx_map, "if_filter_order", source, "rf.receiver", min_value=1
        ),
        if_filter_type=_require_str(rx_map, "if_filter_type", source, "rf.receiver"),
        receiver_gain_db=_require_float(rx_map, "receiver_gain_db", source, "rf.receiver"),
        adc_full_scale_vpp=_require_float(
            rx_map, "adc_full_scale_vpp", source, "rf.receiver", gt=0.0
        ),
        adc_input_impedance_ohm=_require_float(
            rx_map, "adc_input_impedance_ohm", source, "rf.receiver", gt=0.0
        ),
        adc_bits=_require_int(
            rx_map, "adc_bits", source, "rf.receiver", min_value=1, max_value=32
        ),
        adc_clip_enabled=_require_bool(rx_map, "adc_clip_enabled", source, "rf.receiver"),
        adc_quantization_enabled=_require_bool(
            rx_map, "adc_quantization_enabled", source, "rf.receiver"
        ),
        sample_representation=_require_str(
            rx_map, "sample_representation", source, "rf.receiver"
        ),
        receiver_gain_status=_optional_str(
            rx_map, "receiver_gain_status", source, "rf.receiver"
        ),
        adc_bits_status=_optional_str(rx_map, "adc_bits_status", source, "rf.receiver"),
    )

    rf_cfg = RFConfig(
        speed_of_light_m_per_s=c,
        carrier_frequency_hz=fc,
        operating_band_hz=band,
        tx_count=tx_count,
        rx_count=rx_count,
        tx_schedule=tx_schedule,
        tdm_mode=tdm_mode,
        tx_power_dbm_per_active_tx=_require_float(
            rf_map, "tx_power_dbm_per_active_tx", source, "rf"
        ),
        antenna_gain_dbi_peak=_require_float(rf_map, "antenna_gain_dbi_peak", source, "rf"),
        antenna_pattern=antenna_pattern,
        receiver_noise_figure_db=_require_float(
            rf_map, "receiver_noise_figure_db", source, "rf", ge=0.0
        ),
        phase_noise=phase_noise,
        receiver=receiver,
        tx_power_provenance=_optional_str(rf_map, "tx_power_provenance", source, "rf"),
        antenna_gain_provenance=_optional_str(rf_map, "antenna_gain_provenance", source, "rf"),
        receiver_noise_figure_provenance=_optional_str(
            rf_map, "receiver_noise_figure_provenance", source, "rf"
        ),
    )

    # waveform section
    wf_map = _require_mapping(_require_key(root, "waveform", source), source, "waveform")
    wf_start_freq = _require_float(wf_map, "start_frequency_hz", source, "waveform", gt=0.0)
    wf_slope = _require_float(wf_map, "slope_hz_per_s", source, "waveform", gt=0.0)
    wf_ramp_time = _require_float(wf_map, "ramp_time_s", source, "waveform", gt=0.0)
    wf_adc_start = _require_float(wf_map, "adc_start_time_s", source, "waveform", ge=0.0)
    wf_fs = _require_float(wf_map, "adc_sample_rate_hz", source, "waveform", gt=0.0)
    wf_n_adc = _require_int(wf_map, "adc_samples_per_chirp", source, "waveform", min_value=1)
    wf_idle = _require_float(wf_map, "idle_time_s", source, "waveform", ge=0.0)
    wf_chirp_period = _require_float(wf_map, "chirp_period_s", source, "waveform", gt=0.0)
    wf_chirps_per_tx = _require_int(
        wf_map, "chirps_per_tx_per_frame", source, "waveform", min_value=1
    )
    wf_frame_period = _require_float(wf_map, "frame_period_s", source, "waveform", gt=0.0)
    wf_frame_rate = _require_float(wf_map, "frame_rate_hz", source, "waveform", gt=0.0)
    wf_nom_bw = _require_float(
        wf_map, "nominal_sweep_bandwidth_hz", source, "waveform", gt=0.0
    )
    wf_range_fft = _require_int(wf_map, "range_fft_size", source, "waveform", min_value=wf_n_adc)
    wf_doppler_fft = _require_int(
        wf_map, "doppler_fft_size", source, "waveform", min_value=wf_chirps_per_tx
    )
    wf_angle_az_fft = _require_int(
        wf_map, "angle_fft_azimuth_size", source, "waveform", min_value=1
    )
    wf_angle_el_fft = _require_int(
        wf_map, "angle_fft_elevation_size", source, "waveform", min_value=1
    )

    win_map = _require_mapping(
        _require_key(wf_map, "window", source, "waveform"), source, "waveform.window"
    )
    win_range = _require_str(win_map, "range", source, "waveform.window")
    win_doppler = _require_str(win_map, "doppler", source, "waveform.window")
    win_angle = _require_str(win_map, "angle", source, "waveform.window")
    for w_name, w_val in (("range", win_range), ("doppler", win_doppler), ("angle", win_angle)):
        if w_val not in VALID_WINDOWS:
            raise _fail(
                source,
                f"waveform.window.{w_name}",
                f"unsupported window {w_val!r}; expected one of {sorted(VALID_WINDOWS)}",
            )

    # Cross-field RF / waveform physical consistency checks
    if receiver.max_if_frequency_hz >= wf_fs / 2.0:
        raise _fail(
            source,
            "rf.receiver.max_if_frequency_hz",
            f"max_if_frequency_hz ({receiver.max_if_frequency_hz}) must be below "
            f"real-ADC Nyquist frequency ({wf_fs / 2.0})",
        )
    if not (band[0] <= wf_start_freq <= band[1]):
        raise _fail(
            source,
            "waveform.start_frequency_hz",
            f"start_frequency_hz ({wf_start_freq}) must lie within operating_band_hz [{band[0]}, {band[1]}]",
        )
    sweep_end_freq = wf_start_freq + wf_slope * wf_ramp_time
    if sweep_end_freq > band[1] + 1.0:
        raise _fail(
            source,
            "waveform.slope_hz_per_s",
            f"chirp end frequency ({sweep_end_freq:.3e} Hz) exceeds operating_band_hz upper bound ({band[1]:.3e} Hz)",
        )
    adc_end_time = wf_adc_start + wf_n_adc / wf_fs
    if adc_end_time > wf_ramp_time + 1e-12:
        raise _fail(
            source,
            "waveform.adc_samples_per_chirp",
            f"ADC sampling window ends at {adc_end_time:.9g} s, exceeding ramp_time_s ({wf_ramp_time:.9g} s)",
        )
    expected_chirp_period = wf_ramp_time + wf_idle
    if abs(wf_chirp_period - expected_chirp_period) > 1e-9:
        raise _fail(
            source,
            "waveform.chirp_period_s",
            f"chirp_period_s ({wf_chirp_period:.9g}) must equal ramp_time_s + idle_time_s ({expected_chirp_period:.9g})",
        )
    if abs(wf_frame_rate - (1.0 / wf_frame_period)) > 1e-6 * wf_frame_rate:
        raise _fail(
            source,
            "waveform.frame_rate_hz",
            f"frame_rate_hz ({wf_frame_rate}) is inconsistent with 1 / frame_period_s ({1.0 / wf_frame_period:.6g})",
        )
    expected_nom_bw = wf_slope * wf_ramp_time
    if abs(wf_nom_bw - expected_nom_bw) > 1e-6 * expected_nom_bw:
        raise _fail(
            source,
            "waveform.nominal_sweep_bandwidth_hz",
            f"nominal_sweep_bandwidth_hz ({wf_nom_bw:.6g}) must equal slope_hz_per_s * ramp_time_s ({expected_nom_bw:.6g})",
        )
    tx_cycle_length = len(tx_schedule) if tdm_mode else 1
    active_frame_time = wf_chirps_per_tx * tx_cycle_length * wf_chirp_period
    if active_frame_time > wf_frame_period + 1e-12:
        raise _fail(
            source,
            "waveform.frame_period_s",
            f"active chirp sequence duration ({active_frame_time:.6g} s) exceeds frame_period_s ({wf_frame_period:.6g} s)",
        )

    wf_cfg = WaveformConfig(
        type=_require_str(wf_map, "type", source, "waveform"),
        start_frequency_hz=wf_start_freq,
        slope_hz_per_s=wf_slope,
        ramp_time_s=wf_ramp_time,
        adc_start_time_s=wf_adc_start,
        adc_sample_rate_hz=wf_fs,
        adc_samples_per_chirp=wf_n_adc,
        adc_sampling=_require_str(wf_map, "adc_sampling", source, "waveform"),
        idle_time_s=wf_idle,
        chirp_period_s=wf_chirp_period,
        chirps_per_tx_per_frame=wf_chirps_per_tx,
        frame_period_s=wf_frame_period,
        frame_rate_hz=wf_frame_rate,
        nominal_sweep_bandwidth_hz=wf_nom_bw,
        range_fft_size=wf_range_fft,
        doppler_fft_size=wf_doppler_fft,
        angle_fft_azimuth_size=wf_angle_az_fft,
        angle_fft_elevation_size=wf_angle_el_fft,
        window=WindowConfig(range=win_range, doppler=win_doppler, angle=win_angle),
    )

    # antenna_array section
    arr_map = _require_mapping(
        _require_key(root, "antenna_array", source), source, "antenna_array"
    )
    coord_units = _require_str(arr_map, "coordinate_units", source, "antenna_array")
    if coord_units not in VALID_COORDINATE_UNITS:
        raise _fail(
            source,
            "antenna_array.coordinate_units",
            f"unsupported coordinate_units {coord_units!r}; expected one of {sorted(VALID_COORDINATE_UNITS)}",
        )
    raw_rx_pos = _require_list(
        arr_map, "rx_positions_lambda", source, "antenna_array", exact_length=rx_count
    )
    rx_positions = tuple(
        _parse_vec3_item(item, source, f"antenna_array.rx_positions_lambda[{idx}]")
        for idx, item in enumerate(raw_rx_pos)
    )
    raw_tx_pos = _require_list(
        arr_map, "tx_positions_lambda", source, "antenna_array", exact_length=tx_count
    )
    tx_positions = tuple(
        _parse_vec3_item(item, source, f"antenna_array.tx_positions_lambda[{idx}]")
        for idx, item in enumerate(raw_tx_pos)
    )
    array_cfg = AntennaArrayConfig(
        layout=_require_str(arr_map, "layout", source, "antenna_array"),
        coordinate_units=coord_units,
        rx_positions_lambda=rx_positions,
        tx_positions_lambda=tx_positions,
        virtual_position_rule=_require_str(
            arr_map, "virtual_position_rule", source, "antenna_array"
        ),
        channel_gain_mismatch_db_rms=_require_float(
            arr_map, "channel_gain_mismatch_db_rms", source, "antenna_array", ge=0.0
        ),
        channel_phase_mismatch_deg_rms=_require_float(
            arr_map, "channel_phase_mismatch_deg_rms", source, "antenna_array", ge=0.0
        ),
        status=_optional_str(arr_map, "status", source, "antenna_array"),
        channel_mismatch_status=_optional_str(
            arr_map, "channel_mismatch_status", source, "antenna_array"
        ),
    )

    # impairments section
    imp_map = _require_mapping(
        _require_key(root, "impairments", source), source, "impairments"
    )
    imp_pn = _require_mapping(
        _require_key(imp_map, "phase_noise", source, "impairments"),
        source,
        "impairments.phase_noise",
    )
    imp_slope = _require_mapping(
        _require_key(imp_map, "chirp_slope_error", source, "impairments"),
        source,
        "impairments.chirp_slope_error",
    )
    imp_start_f = _require_mapping(
        _require_key(imp_map, "chirp_start_frequency_error", source, "impairments"),
        source,
        "impairments.chirp_start_frequency_error",
    )
    imp_ch = _require_mapping(
        _require_key(imp_map, "tx_rx_channel_mismatch", source, "impairments"),
        source,
        "impairments.tx_rx_channel_mismatch",
    )
    imp_iq = _require_mapping(
        _require_key(imp_map, "iq_imbalance", source, "impairments"),
        source,
        "impairments.iq_imbalance",
    )
    imp_quant = _require_mapping(
        _require_key(imp_map, "adc_quantization", source, "impairments"),
        source,
        "impairments.adc_quantization",
    )
    imp_clip = _require_mapping(
        _require_key(imp_map, "adc_clipping", source, "impairments"),
        source,
        "impairments.adc_clipping",
    )
    imp_leak = _require_mapping(
        _require_key(imp_map, "tx_leakage", source, "impairments"),
        source,
        "impairments.tx_leakage",
    )
    imp_jitter = _require_mapping(
        _require_key(imp_map, "timing_jitter", source, "impairments"),
        source,
        "impairments.timing_jitter",
    )
    imp_thermal = _require_mapping(
        _require_key(imp_map, "thermal_noise", source, "impairments"),
        source,
        "impairments.thermal_noise",
    )

    impairments_cfg = ImpairmentsConfig(
        phase_noise_enabled=_require_bool(imp_pn, "enabled", source, "impairments.phase_noise"),
        phase_noise_strength_scale=_require_float(
            imp_pn, "strength_scale", source, "impairments.phase_noise", ge=0.0
        ),
        chirp_slope_error_enabled=_require_bool(
            imp_slope, "enabled", source, "impairments.chirp_slope_error"
        ),
        chirp_slope_error_sigma_ppm=_require_float(
            imp_slope, "sigma_ppm", source, "impairments.chirp_slope_error", ge=0.0
        ),
        chirp_start_frequency_error_enabled=_require_bool(
            imp_start_f, "enabled", source, "impairments.chirp_start_frequency_error"
        ),
        chirp_start_frequency_error_sigma_hz=_require_float(
            imp_start_f, "sigma_hz", source, "impairments.chirp_start_frequency_error", ge=0.0
        ),
        tx_rx_channel_mismatch_enabled=_require_bool(
            imp_ch, "enabled", source, "impairments.tx_rx_channel_mismatch"
        ),
        iq_imbalance_enabled=_require_bool(imp_iq, "enabled", source, "impairments.iq_imbalance"),
        iq_imbalance_amplitude_mismatch_db_sigma=_require_float(
            imp_iq, "amplitude_mismatch_db_sigma", source, "impairments.iq_imbalance", ge=0.0
        ),
        iq_imbalance_phase_mismatch_deg_sigma=_require_float(
            imp_iq, "phase_mismatch_deg_sigma", source, "impairments.iq_imbalance", ge=0.0
        ),
        adc_quantization_enabled=_require_bool(
            imp_quant, "enabled", source, "impairments.adc_quantization"
        ),
        adc_clipping_enabled=_require_bool(imp_clip, "enabled", source, "impairments.adc_clipping"),
        tx_leakage_enabled=_require_bool(imp_leak, "enabled", source, "impairments.tx_leakage"),
        tx_leakage_level_db_relative_to_full_scale=_require_float(
            imp_leak,
            "leakage_level_db_relative_to_full_scale",
            source,
            "impairments.tx_leakage",
            le=0.0,
        ),
        timing_jitter_enabled=_require_bool(
            imp_jitter, "enabled", source, "impairments.timing_jitter"
        ),
        timing_jitter_sigma_s=_require_float(
            imp_jitter, "sigma_s", source, "impairments.timing_jitter", ge=0.0
        ),
        thermal_noise_enabled=_require_bool(
            imp_thermal, "enabled", source, "impairments.thermal_noise"
        ),
        provenance=_optional_str(imp_map, "impairments_provenance", source, "impairments"),
    )

    # human_scattering section
    hs_map = _require_mapping(
        _require_key(root, "human_scattering", source), source, "human_scattering"
    )
    rcs_map = _require_mapping(
        _require_key(hs_map, "rcs_prior_dbsm", source, "human_scattering"),
        source,
        "human_scattering.rcs_prior_dbsm",
    )
    rcs_min = _require_float(rcs_map, "minimum", source, "human_scattering.rcs_prior_dbsm")
    rcs_nom = _require_float(rcs_map, "nominal", source, "human_scattering.rcs_prior_dbsm")
    rcs_max = _require_float(rcs_map, "maximum", source, "human_scattering.rcs_prior_dbsm")
    if not (rcs_min <= rcs_nom <= rcs_max):
        raise _fail(
            source,
            "human_scattering.rcs_prior_dbsm",
            f"expected minimum <= nominal <= maximum, got ({rcs_min}, {rcs_nom}, {rcs_max})",
        )
    human_scattering = HumanScatteringConfig(
        model=_require_str(hs_map, "model", source, "human_scattering"),
        rcs_prior_dbsm=RCSPriorConfig(minimum=rcs_min, nominal=rcs_nom, maximum=rcs_max),
        aspect_dependence_enabled=_require_bool(
            hs_map, "aspect_dependence_enabled", source, "human_scattering"
        ),
        clothing_scattering_scale=_require_float(
            hs_map, "clothing_scattering_scale", source, "human_scattering", gt=0.0
        ),
        deterministic_micro_scatterers_per_body_segment=_require_int(
            hs_map,
            "deterministic_micro_scatterers_per_body_segment",
            source,
            "human_scattering",
            min_value=0,
        ),
        random_scatterers_per_body_segment=_require_int(
            hs_map,
            "random_scatterers_per_body_segment",
            source,
            "human_scattering",
            min_value=0,
        ),
        rcs_status=_optional_str(hs_map, "rcs_status", source, "human_scattering"),
        note=_optional_str(hs_map, "note", source, "human_scattering"),
    )

    radar_profile = RadarProfile(
        schema_version=schema_version,
        profile=profile_meta,
        rf=rf_cfg,
        waveform=wf_cfg,
        antenna_array=array_cfg,
        impairments=impairments_cfg,
        human_scattering=human_scattering,
    )

    expected_derived: ExpectedDerivedMetrics | None = None
    if "expected_derived" in root:
        expected_derived = parse_expected_derived(root, source=source)
        if validate_expected:
            actual = compute_derived_metrics(radar_profile)
            validate_derived_metrics(actual, expected_derived, source=source)
    elif require_expected_derived:
        raise ConfigValidationError(
            "Missing required field 'expected_derived'",
            source=source,
        )

    return radar_profile, expected_derived


def _parse_vec3_item(item: Any, source: str, path: str) -> tuple[float, float, float]:
    if not isinstance(item, list) or len(item) != 3:
        raise _fail(source, path, f"expected a 3-element coordinate list, got {item!r}")
    return (
        _as_float(item[0], source, f"{path}[0]"),
        _as_float(item[1], source, f"{path}[1]"),
        _as_float(item[2], source, f"{path}[2]"),
    )


def load_radar_config_with_expected(
    path: str | Path,
    *,
    require_expected_derived: bool = False,
    validate_expected: bool = True,
) -> tuple[RadarProfile, ExpectedDerivedMetrics | None]:
    file_path = Path(path)
    raw = _load_yaml_mapping(file_path)
    return parse_radar_profile(
        raw,
        source=file_path.name,
        require_expected_derived=require_expected_derived,
        validate_expected=validate_expected,
    )


def load_radar_profile(
    path: str | Path,
    *,
    require_expected_derived: bool = False,
    validate_expected: bool = True,
) -> RadarProfile:
    profile, _ = load_radar_config_with_expected(
        path,
        require_expected_derived=require_expected_derived,
        validate_expected=validate_expected,
    )
    return profile


# ---------------------------------------------------------------------------
# Material database (material_db.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConductivityModel:
    type: str
    coefficient: float
    exponent: float
    frequency_unit: str
    equation: str | None = None


@dataclass(frozen=True)
class MaterialEntry:
    material_id: str
    model: str
    epsilon_real: float
    epsilon_imag: float | None
    conductivity_model: ConductivityModel | None
    roughness_rms_m: float
    thickness_default_m: float | None
    valid_frequency_hz: tuple[float, float] | None
    provenance: str
    source_url: str | None = None
    uncertainty: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class MaterialAlias:
    material_id: str
    alias_of: str
    provenance: str | None = None


@dataclass(frozen=True)
class SolverConventions:
    fresnel_polarizations: tuple[str, ...]
    incidence_angle: str | None = None
    conductivity_loss: str | None = None
    diffuse_scattering: str | None = None


@dataclass(frozen=True)
class MaterialDatabase:
    schema_version: str
    reference_frequency_hz: float
    materials: dict[str, MaterialEntry]
    aliases: dict[str, MaterialAlias]
    solver_conventions: SolverConventions
    notes: str | None = None

    def has_material(self, material_id: str) -> bool:
        return material_id in self.materials or material_id in self.aliases

    def resolve(self, material_id: str) -> MaterialEntry:
        current = material_id
        visited: set[str] = set()
        while current in self.aliases:
            if current in visited:
                raise ConfigValidationError(
                    f"Cyclic material alias detected involving {material_id!r}",
                    field_path=f"materials.{material_id}",
                )
            visited.add(current)
            current = self.aliases[current].alias_of
        if current not in self.materials:
            raise ConfigValidationError(
                f"Unknown material_id {material_id!r}",
                field_path="materials",
            )
        return self.materials[current]

    def all_ids(self) -> tuple[str, ...]:
        return tuple(sorted((*self.materials.keys(), *self.aliases.keys())))


def parse_material_db(
    raw: Mapping[str, Any], *, source: str = "material_db.yaml"
) -> MaterialDatabase:
    root = _require_mapping(raw, source, "<root>")
    schema_version = _require_str(root, "schema_version", source)
    ref_freq = _require_float(root, "reference_frequency_hz", source, gt=0.0)
    notes = _optional_str(root, "notes", source)

    mat_map = _require_mapping(_require_key(root, "materials", source), source, "materials")
    if len(mat_map) == 0:
        raise _fail(source, "materials", "must define at least one material")

    materials: dict[str, MaterialEntry] = {}
    aliases: dict[str, MaterialAlias] = {}

    for mat_id, entry_raw in mat_map.items():
        if not isinstance(mat_id, str) or not mat_id.strip():
            raise _fail(source, "materials", f"material key must be a non-empty string, got {mat_id!r}")
        entry_path = f"materials.{mat_id}"
        entry = _require_mapping(entry_raw, source, entry_path)
        if "alias_of" in entry:
            target = _require_str(entry, "alias_of", source, entry_path)
            aliases[mat_id] = MaterialAlias(
                material_id=mat_id,
                alias_of=target,
                provenance=_optional_str(entry, "provenance", source, entry_path),
            )
            continue

        model = _require_str(entry, "model", source, entry_path)
        eps_real = _require_float(entry, "epsilon_real", source, entry_path, ge=1.0)
        eps_imag = _optional_float(entry, "epsilon_imag", source, entry_path, ge=0.0)

        cond_model: ConductivityModel | None = None
        if "conductivity_model" in entry and entry["conductivity_model"] is not None:
            cond_path = f"{entry_path}.conductivity_model"
            cond_map = _require_mapping(entry["conductivity_model"], source, cond_path)
            cond_model = ConductivityModel(
                type=_require_str(cond_map, "type", source, cond_path),
                coefficient=_require_float(cond_map, "coefficient", source, cond_path, gt=0.0),
                exponent=_require_float(cond_map, "exponent", source, cond_path),
                frequency_unit=_require_str(cond_map, "frequency_unit", source, cond_path),
                equation=_optional_str(cond_map, "equation", source, cond_path),
            )

        if eps_imag is None and cond_model is None:
            raise _fail(
                source,
                entry_path,
                "material must specify either 'epsilon_imag' or 'conductivity_model'",
            )
        if eps_imag is not None and eps_imag > 0.0 and cond_model is not None:
            raise _fail(
                source,
                entry_path,
                "material specifies both non-zero 'epsilon_imag' and 'conductivity_model' "
                "(risk of double-counting dielectric and conductivity loss)",
            )

        roughness = _require_float(entry, "roughness_rms_m", source, entry_path, ge=0.0)
        thickness = _optional_float(entry, "thickness_default_m", source, entry_path, gt=0.0)
        valid_freq: tuple[float, float] | None = None
        if "valid_frequency_hz" in entry and entry["valid_frequency_hz"] is not None:
            valid_freq = _require_range_pair(
                entry, "valid_frequency_hz", source, entry_path, gt=0.0
            )
        provenance = _require_str(entry, "provenance", source, entry_path)
        materials[mat_id] = MaterialEntry(
            material_id=mat_id,
            model=model,
            epsilon_real=eps_real,
            epsilon_imag=eps_imag,
            conductivity_model=cond_model,
            roughness_rms_m=roughness,
            thickness_default_m=thickness,
            valid_frequency_hz=valid_freq,
            provenance=provenance,
            source_url=_optional_str(entry, "source_url", source, entry_path),
            uncertainty=_optional_str(entry, "uncertainty", source, entry_path),
            notes=_optional_str(entry, "notes", source, entry_path),
        )

    # Validate alias targets and acyclicity
    for alias_id, alias_entry in aliases.items():
        current = alias_entry.alias_of
        seen = {alias_id}
        while current in aliases:
            if current in seen:
                raise _fail(
                    source,
                    f"materials.{alias_id}.alias_of",
                    f"cyclic material alias chain involving {alias_id!r}",
                )
            seen.add(current)
            current = aliases[current].alias_of
        if current not in materials:
            raise _fail(
                source,
                f"materials.{alias_id}.alias_of",
                f"alias target {alias_entry.alias_of!r} does not exist in materials",
            )

    conv_map = _require_mapping(
        _require_key(root, "solver_conventions", source), source, "solver_conventions"
    )
    raw_pols = _require_list(
        conv_map, "fresnel_polarizations", source, "solver_conventions", non_empty=True
    )
    pols: list[str] = []
    for idx, pol in enumerate(raw_pols):
        if not isinstance(pol, str) or pol not in VALID_POLARIZATIONS:
            raise _fail(
                source,
                f"solver_conventions.fresnel_polarizations[{idx}]",
                f"unsupported polarization {pol!r}; expected one of {sorted(VALID_POLARIZATIONS)}",
            )
        pols.append(pol)

    conventions = SolverConventions(
        fresnel_polarizations=tuple(pols),
        incidence_angle=_optional_str(conv_map, "incidence_angle", source, "solver_conventions"),
        conductivity_loss=_optional_str(
            conv_map, "conductivity_loss", source, "solver_conventions"
        ),
        diffuse_scattering=_optional_str(
            conv_map, "diffuse_scattering", source, "solver_conventions"
        ),
    )

    return MaterialDatabase(
        schema_version=schema_version,
        reference_frequency_hz=ref_freq,
        materials=materials,
        aliases=aliases,
        solver_conventions=conventions,
        notes=notes,
    )


def load_material_db(path: str | Path) -> MaterialDatabase:
    file_path = Path(path)
    raw = _load_yaml_mapping(file_path)
    return parse_material_db(raw, source=file_path.name)


# ---------------------------------------------------------------------------
# Scene configuration (scene.json)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialLayer:
    material_id: str
    thickness_m: float


@dataclass(frozen=True)
class MetalStudsConfig:
    enabled: bool
    spacing_m: float
    status: str | None = None


@dataclass(frozen=True)
class SurfaceConstruction:
    construction_id: str
    layers: tuple[MaterialLayer, ...]
    metal_studs: MetalStudsConfig | None = None
    thickness_status: str | None = None
    source_note: str | None = None


@dataclass(frozen=True)
class RoomConstruction:
    walls: SurfaceConstruction | None = None
    floor: SurfaceConstruction | None = None
    ceiling: SurfaceConstruction | None = None


@dataclass(frozen=True)
class RoomConfig:
    width_x_m: float
    depth_y_m: float
    height_z_m: float
    wall_thickness_m: float
    floor_material_id: str
    wall_material_id: str
    ceiling_material_id: str
    construction: RoomConstruction | None = None


@dataclass(frozen=True)
class OpeningConfig:
    id: str
    kind: str
    wall: str
    center_m: tuple[float, float, float]
    width_m: float
    height_m: float
    sill_height_m: float
    material_id: str
    hinge_side: str | None = None
    swing_direction: str | None = None
    open_fraction: float | None = None
    construction_layers: tuple[MaterialLayer, ...] = ()


@dataclass(frozen=True)
class SceneObjectConfig:
    id: str
    kind: str
    position_m: tuple[float, float, float]
    dimensions_m: tuple[float, float, float]
    yaw_rad: float
    material_id: str
    component_materials: dict[str, str]
    collision_shape: str
    notes: str | None = None


@dataclass(frozen=True)
class RadarMountConfig:
    id: str
    profile_id: str
    mount_type: str
    position_m: tuple[float, float, float]
    boresight_unit_vector_world: tuple[float, float, float]
    yaw_rad: float
    tilt_down_deg: float
    wall: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class InitialActorConfig:
    actor_id: str
    avatar_asset_id: str
    start_position_m: tuple[float, float, float]
    entry_target_m: tuple[float, float, float]


@dataclass(frozen=True)
class SceneConfig:
    schema_version: str
    scene_id: str
    units: str
    coordinate_frame: str | None
    room: RoomConfig
    openings: tuple[OpeningConfig, ...]
    objects: tuple[SceneObjectConfig, ...]
    radar: RadarMountConfig
    initial_actor: InitialActorConfig | None = None

    def referenced_material_ids(self) -> set[str]:
        ids: set[str] = {
            self.room.floor_material_id,
            self.room.wall_material_id,
            self.room.ceiling_material_id,
        }
        if self.room.construction is not None:
            for surface in (
                self.room.construction.walls,
                self.room.construction.floor,
                self.room.construction.ceiling,
            ):
                if surface is not None:
                    ids.update(layer.material_id for layer in surface.layers)
        for opening in self.openings:
            ids.add(opening.material_id)
            ids.update(layer.material_id for layer in opening.construction_layers)
        for obj in self.objects:
            ids.add(obj.material_id)
            ids.update(obj.component_materials.values())
        return ids


def _parse_material_layers(
    raw_layers: Any, source: str, path: str
) -> tuple[MaterialLayer, ...]:
    if not isinstance(raw_layers, list) or len(raw_layers) == 0:
        raise _fail(source, path, "expected a non-empty list of material layers")
    layers: list[MaterialLayer] = []
    for idx, item in enumerate(raw_layers):
        item_path = f"{path}[{idx}]"
        layer_map = _require_mapping(item, source, item_path)
        layers.append(
            MaterialLayer(
                material_id=_require_str(layer_map, "material_id", source, item_path),
                thickness_m=_require_float(layer_map, "thickness_m", source, item_path, gt=0.0),
            )
        )
    return tuple(layers)


def _parse_surface_construction(
    raw: Mapping[str, Any], source: str, path: str, *, layer_key: str
) -> SurfaceConstruction:
    cmap = _require_mapping(raw, source, path)
    layers_raw = _require_key(cmap, layer_key, source, path)
    layers = _parse_material_layers(layers_raw, source, _join_path(path, layer_key))
    metal_studs: MetalStudsConfig | None = None
    if "metal_studs" in cmap and cmap["metal_studs"] is not None:
        studs_path = f"{path}.metal_studs"
        studs_map = _require_mapping(cmap["metal_studs"], source, studs_path)
        metal_studs = MetalStudsConfig(
            enabled=_require_bool(studs_map, "enabled", source, studs_path),
            spacing_m=_require_float(studs_map, "spacing_m", source, studs_path, gt=0.0),
            status=_optional_str(studs_map, "status", source, studs_path),
        )
    return SurfaceConstruction(
        construction_id=_require_str(cmap, "construction_id", source, path),
        layers=layers,
        metal_studs=metal_studs,
        thickness_status=_optional_str(cmap, "thickness_status", source, path),
        source_note=_optional_str(cmap, "source_note", source, path),
    )


def parse_scene_config(
    raw: Mapping[str, Any], *, source: str = "scene.json"
) -> SceneConfig:
    root = _require_mapping(raw, source, "<root>")
    schema_version = _require_str(root, "schema_version", source)
    scene_id = _require_str(root, "scene_id", source)
    units = _require_str(root, "units", source)
    if units != "m":
        raise _fail(source, "units", f"expected SI units 'm', got {units!r}")
    coord_frame = _optional_str(root, "coordinate_frame", source)

    room_map = _require_mapping(_require_key(root, "room", source), source, "room")
    width_x = _require_float(room_map, "width_x_m", source, "room", gt=0.0)
    depth_y = _require_float(room_map, "depth_y_m", source, "room", gt=0.0)
    height_z = _require_float(room_map, "height_z_m", source, "room", gt=0.0)
    wall_thickness = _require_float(room_map, "wall_thickness_m", source, "room", gt=0.0)

    construction: RoomConstruction | None = None
    if "construction" in room_map and room_map["construction"] is not None:
        c_map = _require_mapping(room_map["construction"], source, "room.construction")
        walls_c = (
            _parse_surface_construction(
                c_map["walls"],
                source,
                "room.construction.walls",
                layer_key="layers_from_room_to_outside",
            )
            if "walls" in c_map and c_map["walls"] is not None
            else None
        )
        floor_c = (
            _parse_surface_construction(
                c_map["floor"], source, "room.construction.floor", layer_key="layers"
            )
            if "floor" in c_map and c_map["floor"] is not None
            else None
        )
        ceiling_c = (
            _parse_surface_construction(
                c_map["ceiling"], source, "room.construction.ceiling", layer_key="layers"
            )
            if "ceiling" in c_map and c_map["ceiling"] is not None
            else None
        )
        construction = RoomConstruction(walls=walls_c, floor=floor_c, ceiling=ceiling_c)

    room_cfg = RoomConfig(
        width_x_m=width_x,
        depth_y_m=depth_y,
        height_z_m=height_z,
        wall_thickness_m=wall_thickness,
        floor_material_id=_require_str(room_map, "floor_material_id", source, "room"),
        wall_material_id=_require_str(room_map, "wall_material_id", source, "room"),
        ceiling_material_id=_require_str(room_map, "ceiling_material_id", source, "room"),
        construction=construction,
    )

    # openings
    raw_openings = _require_list(root, "openings", source)
    openings: list[OpeningConfig] = []
    seen_opening_ids: set[str] = set()
    for idx, item in enumerate(raw_openings):
        o_path = f"openings[{idx}]"
        o_map = _require_mapping(item, source, o_path)
        o_id = _require_str(o_map, "id", source, o_path)
        if o_id in seen_opening_ids:
            raise _fail(source, f"{o_path}.id", f"duplicate opening id {o_id!r}")
        seen_opening_ids.add(o_id)
        kind = _require_str(o_map, "kind", source, o_path)
        if kind not in VALID_OPENING_KINDS:
            raise _fail(
                source,
                f"{o_path}.kind",
                f"unsupported opening kind {kind!r}; expected one of {sorted(VALID_OPENING_KINDS)}",
            )
        wall = _require_str(o_map, "wall", source, o_path)
        if wall not in VALID_WALLS:
            raise _fail(
                source,
                f"{o_path}.wall",
                f"unsupported wall {wall!r}; expected one of {sorted(VALID_WALLS)}",
            )
        o_width = _require_float(o_map, "width_m", source, o_path, gt=0.0)
        o_height = _require_float(o_map, "height_m", source, o_path, gt=0.0)
        sill = _require_float(o_map, "sill_height_m", source, o_path, ge=0.0)
        if sill + o_height > height_z + 1e-9:
            raise _fail(
                source,
                f"{o_path}.height_m",
                f"opening top ({sill + o_height:.4g} m) exceeds room height ({height_z:.4g} m)",
            )
        open_fraction = _optional_float(
            o_map, "open_fraction", source, o_path, ge=0.0, le=1.0
        )
        layers = (
            _parse_material_layers(
                o_map["construction_layers"], source, f"{o_path}.construction_layers"
            )
            if "construction_layers" in o_map and o_map["construction_layers"] is not None
            else ()
        )
        openings.append(
            OpeningConfig(
                id=o_id,
                kind=kind,
                wall=wall,
                center_m=_require_vec3(o_map, "center_m", source, o_path),
                width_m=o_width,
                height_m=o_height,
                sill_height_m=sill,
                material_id=_require_str(o_map, "material_id", source, o_path),
                hinge_side=_optional_str(o_map, "hinge_side", source, o_path),
                swing_direction=_optional_str(o_map, "swing_direction", source, o_path),
                open_fraction=open_fraction,
                construction_layers=layers,
            )
        )

    # furniture objects
    raw_objects = _require_list(root, "objects", source)
    objects: list[SceneObjectConfig] = []
    seen_object_ids: set[str] = set()
    for idx, item in enumerate(raw_objects):
        obj_path = f"objects[{idx}]"
        obj_map = _require_mapping(item, source, obj_path)
        obj_id = _require_str(obj_map, "id", source, obj_path)
        if obj_id in seen_object_ids:
            raise _fail(source, f"{obj_path}.id", f"duplicate object id {obj_id!r}")
        seen_object_ids.add(obj_id)

        comp_mats: dict[str, str] = {}
        if "component_materials" in obj_map and obj_map["component_materials"] is not None:
            cm_map = _require_mapping(
                obj_map["component_materials"], source, f"{obj_path}.component_materials"
            )
            for comp_name, mat_name in cm_map.items():
                if not isinstance(comp_name, str) or not comp_name.strip():
                    raise _fail(
                        source,
                        f"{obj_path}.component_materials",
                        "component names must be non-empty strings",
                    )
                if not isinstance(mat_name, str) or not mat_name.strip():
                    raise _fail(
                        source,
                        f"{obj_path}.component_materials.{comp_name}",
                        "material_id must be a non-empty string",
                    )
                comp_mats[comp_name] = mat_name

        objects.append(
            SceneObjectConfig(
                id=obj_id,
                kind=_require_str(obj_map, "kind", source, obj_path),
                position_m=_require_vec3(obj_map, "position_m", source, obj_path),
                dimensions_m=_require_vec3(
                    obj_map, "dimensions_m", source, obj_path, positive=True
                ),
                yaw_rad=_require_float(obj_map, "yaw_rad", source, obj_path),
                material_id=_require_str(obj_map, "material_id", source, obj_path),
                component_materials=comp_mats,
                collision_shape=_require_str(obj_map, "collision_shape", source, obj_path),
                notes=_optional_str(obj_map, "notes", source, obj_path),
            )
        )

    # radar mount
    rad_map = _require_mapping(_require_key(root, "radar", source), source, "radar")
    rad_pos = _require_vec3(rad_map, "position_m", source, "radar")
    if not (
        0.0 <= rad_pos[0] <= width_x
        and 0.0 <= rad_pos[1] <= depth_y
        and 0.0 <= rad_pos[2] <= height_z
    ):
        raise _fail(
            source,
            "radar.position_m",
            f"radar position {rad_pos} must lie inside room bounds "
            f"[0..{width_x}, 0..{depth_y}, 0..{height_z}]",
        )
    boresight = _require_vec3(rad_map, "boresight_unit_vector_world", source, "radar")
    boresight_norm = math.sqrt(boresight[0] ** 2 + boresight[1] ** 2 + boresight[2] ** 2)
    if abs(boresight_norm - 1.0) > 1e-3:
        raise _fail(
            source,
            "radar.boresight_unit_vector_world",
            f"boresight vector must be unit length (norm=1.0), got norm={boresight_norm:.6f}",
        )
    rad_wall = _optional_str(rad_map, "wall", source, "radar")
    if rad_wall is not None and rad_wall not in VALID_WALLS:
        raise _fail(
            source,
            "radar.wall",
            f"unsupported wall {rad_wall!r}; expected one of {sorted(VALID_WALLS)}",
        )

    radar_mount = RadarMountConfig(
        id=_require_str(rad_map, "id", source, "radar"),
        profile_id=_require_str(rad_map, "profile_id", source, "radar"),
        mount_type=_require_str(rad_map, "mount_type", source, "radar"),
        position_m=rad_pos,
        boresight_unit_vector_world=boresight,
        yaw_rad=_require_float(rad_map, "yaw_rad", source, "radar"),
        tilt_down_deg=_require_float(
            rad_map, "tilt_down_deg", source, "radar", ge=-90.0, le=90.0
        ),
        wall=rad_wall,
        notes=_optional_str(rad_map, "notes", source, "radar"),
    )

    initial_actor: InitialActorConfig | None = None
    if "initial_actor" in root and root["initial_actor"] is not None:
        ia_map = _require_mapping(root["initial_actor"], source, "initial_actor")
        initial_actor = InitialActorConfig(
            actor_id=_require_str(ia_map, "actor_id", source, "initial_actor"),
            avatar_asset_id=_require_str(ia_map, "avatar_asset_id", source, "initial_actor"),
            start_position_m=_require_vec3(ia_map, "start_position_m", source, "initial_actor"),
            entry_target_m=_require_vec3(ia_map, "entry_target_m", source, "initial_actor"),
        )

    return SceneConfig(
        schema_version=schema_version,
        scene_id=scene_id,
        units=units,
        coordinate_frame=coord_frame,
        room=room_cfg,
        openings=tuple(openings),
        objects=tuple(objects),
        radar=radar_mount,
        initial_actor=initial_actor,
    )


def load_scene_config(path: str | Path) -> SceneConfig:
    file_path = Path(path)
    raw = _load_json_mapping(file_path)
    return parse_scene_config(raw, source=file_path.name)


# ---------------------------------------------------------------------------
# Motion configuration (motion_config.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActorConfig:
    actor_id: str
    presentation: str
    height_m_range: tuple[float, float]
    body_scale_range: tuple[float, float]
    motion_sample_rate_hz: float
    visible_avatar_required: bool
    draw_skeleton: bool
    internal_joint_count: int
    collision_body_model: str
    avatar_mesh_style: str
    avatar_asset_policy: str


@dataclass(frozen=True)
class ScenarioModeConfig:
    name: str
    fall_probability: float
    no_fall_probability: float | None = None
    exactly_one_fall: bool = False


@dataclass(frozen=True)
class ScenarioConfig:
    default_mode: str
    modes: dict[str, ScenarioModeConfig]
    random_seed: int
    entry_required: bool
    start_outside_door: bool
    action_sequence_randomized: bool
    minimum_actions_per_run: int
    maximum_actions_per_run: int
    allow_sleep: bool
    allow_intentional_lying_down: bool


@dataclass(frozen=True)
class MotionParametersConfig:
    walking_speed_mps_range: tuple[float, float]
    slow_walk_speed_mps_range: tuple[float, float]
    body_turn_rate_rad_per_s_range: tuple[float, float]
    pause_duration_s_range: tuple[float, float]
    sit_duration_s_range: tuple[float, float]
    sleep_duration_s_range: tuple[float, float]
    lie_down_duration_s_range: tuple[float, float]
    fall_duration_s_range: tuple[float, float]
    fall_recovery_probability: float
    path_planner: str
    navigation_grid_resolution_m: float
    obstacle_clearance_m: float
    full_body_collision_check: bool
    floor_contact_required: bool
    furniture_contact_allowed: tuple[str, ...]


@dataclass(frozen=True)
class FallModelConfig:
    phases: tuple[str, ...]
    use_rigid_body_or_contact_aware_trajectory: bool
    do_not_teleport_pose: bool
    randomize_initial_balance_direction: bool
    ground_truth_event_timestamp_required: bool
    impact_acceleration_status: str


@dataclass(frozen=True)
class SeedingConfig:
    global_seed_source: str
    independent_streams: tuple[str, ...]
    derivation: str


@dataclass(frozen=True)
class MotionConfig:
    schema_version: str
    actor: ActorConfig
    scenario: ScenarioConfig
    motion: MotionParametersConfig
    weighted_actions: dict[str, float]
    hard_negative_actions: tuple[str, ...]
    fall_model: FallModelConfig
    seeding: SeedingConfig


def parse_motion_config(
    raw: Mapping[str, Any], *, source: str = "motion_config.yaml"
) -> MotionConfig:
    root = _require_mapping(raw, source, "<root>")
    schema_version = _require_str(root, "schema_version", source)

    # actor
    actor_map = _require_mapping(_require_key(root, "actor", source), source, "actor")
    visible_avatar = _require_bool(actor_map, "visible_avatar_required", source, "actor")
    if not visible_avatar:
        raise _fail(
            source,
            "actor.visible_avatar_required",
            "visible_avatar_required must be true (full visible human avatar is mandatory)",
        )
    draw_skeleton = _require_bool(actor_map, "draw_skeleton", source, "actor")
    if draw_skeleton:
        raise _fail(
            source,
            "actor.draw_skeleton",
            "draw_skeleton must be false (internal skeleton must remain invisible)",
        )
    joint_count = _require_int(actor_map, "internal_joint_count", source, "actor", min_value=1)
    if joint_count != 17:
        raise _fail(
            source,
            "actor.internal_joint_count",
            f"internal_joint_count must be 17 to match INTERFACES.md, got {joint_count}",
        )
    actor_cfg = ActorConfig(
        actor_id=_require_str(actor_map, "actor_id", source, "actor"),
        presentation=_require_str(actor_map, "presentation", source, "actor"),
        height_m_range=_require_range_pair(actor_map, "height_m_range", source, "actor", gt=0.0),
        body_scale_range=_require_range_pair(
            actor_map, "body_scale_range", source, "actor", gt=0.0
        ),
        motion_sample_rate_hz=_require_float(
            actor_map, "motion_sample_rate_hz", source, "actor", gt=0.0
        ),
        visible_avatar_required=visible_avatar,
        draw_skeleton=draw_skeleton,
        internal_joint_count=joint_count,
        collision_body_model=_require_str(actor_map, "collision_body_model", source, "actor"),
        avatar_mesh_style=_require_str(actor_map, "avatar_mesh_style", source, "actor"),
        avatar_asset_policy=_require_str(actor_map, "avatar_asset_policy", source, "actor"),
    )

    # scenario
    scen_map = _require_mapping(_require_key(root, "scenario", source), source, "scenario")
    modes_map = _require_mapping(
        _require_key(scen_map, "modes", source, "scenario"), source, "scenario.modes"
    )
    if len(modes_map) == 0:
        raise _fail(source, "scenario.modes", "must define at least one scenario mode")
    modes: dict[str, ScenarioModeConfig] = {}
    for mode_name, mode_raw in modes_map.items():
        if not isinstance(mode_name, str) or not mode_name.strip():
            raise _fail(source, "scenario.modes", "mode name must be a non-empty string")
        m_path = f"scenario.modes.{mode_name}"
        m_map = _require_mapping(mode_raw, source, m_path)
        fall_p = _require_float(m_map, "fall_probability", source, m_path, ge=0.0, le=1.0)
        no_fall_p = _optional_float(m_map, "no_fall_probability", source, m_path, ge=0.0, le=1.0)
        if no_fall_p is not None and abs((fall_p + no_fall_p) - 1.0) > 1e-6:
            raise _fail(
                source,
                m_path,
                f"fall_probability ({fall_p}) + no_fall_probability ({no_fall_p}) must equal 1.0",
            )
        exactly_one = _optional_bool(m_map, "exactly_one_fall", source, m_path, default=False)
        modes[mode_name] = ScenarioModeConfig(
            name=mode_name,
            fall_probability=fall_p,
            no_fall_probability=no_fall_p,
            exactly_one_fall=bool(exactly_one),
        )

    default_mode = _require_str(scen_map, "default_mode", source, "scenario")
    if default_mode not in modes:
        raise _fail(
            source,
            "scenario.default_mode",
            f"default_mode {default_mode!r} is not defined in scenario.modes ({sorted(modes)})",
        )
    min_actions = _require_int(scen_map, "minimum_actions_per_run", source, "scenario", min_value=1)
    max_actions = _require_int(
        scen_map, "maximum_actions_per_run", source, "scenario", min_value=min_actions
    )
    scenario_cfg = ScenarioConfig(
        default_mode=default_mode,
        modes=modes,
        random_seed=_require_int(scen_map, "random_seed", source, "scenario"),
        entry_required=_require_bool(scen_map, "entry_required", source, "scenario"),
        start_outside_door=_require_bool(scen_map, "start_outside_door", source, "scenario"),
        action_sequence_randomized=_require_bool(
            scen_map, "action_sequence_randomized", source, "scenario"
        ),
        minimum_actions_per_run=min_actions,
        maximum_actions_per_run=max_actions,
        allow_sleep=_require_bool(scen_map, "allow_sleep", source, "scenario"),
        allow_intentional_lying_down=_require_bool(
            scen_map, "allow_intentional_lying_down", source, "scenario"
        ),
    )

    # motion parameters
    m_map = _require_mapping(_require_key(root, "motion", source), source, "motion")
    raw_contacts = _require_list(
        m_map, "furniture_contact_allowed", source, "motion", non_empty=True
    )
    contacts = tuple(
        _require_non_empty_str_item(item, source, f"motion.furniture_contact_allowed[{idx}]")
        for idx, item in enumerate(raw_contacts)
    )
    motion_params = MotionParametersConfig(
        walking_speed_mps_range=_require_range_pair(
            m_map, "walking_speed_mps_range", source, "motion", gt=0.0
        ),
        slow_walk_speed_mps_range=_require_range_pair(
            m_map, "slow_walk_speed_mps_range", source, "motion", gt=0.0
        ),
        body_turn_rate_rad_per_s_range=_require_range_pair(
            m_map, "body_turn_rate_rad_per_s_range", source, "motion", gt=0.0
        ),
        pause_duration_s_range=_require_range_pair(
            m_map, "pause_duration_s_range", source, "motion", ge=0.0
        ),
        sit_duration_s_range=_require_range_pair(
            m_map, "sit_duration_s_range", source, "motion", gt=0.0
        ),
        sleep_duration_s_range=_require_range_pair(
            m_map, "sleep_duration_s_range", source, "motion", gt=0.0
        ),
        lie_down_duration_s_range=_require_range_pair(
            m_map, "lie_down_duration_s_range", source, "motion", gt=0.0
        ),
        fall_duration_s_range=_require_range_pair(
            m_map, "fall_duration_s_range", source, "motion", gt=0.0
        ),
        fall_recovery_probability=_require_float(
            m_map, "fall_recovery_probability", source, "motion", ge=0.0, le=1.0
        ),
        path_planner=_require_str(m_map, "path_planner", source, "motion"),
        navigation_grid_resolution_m=_require_float(
            m_map, "navigation_grid_resolution_m", source, "motion", gt=0.0
        ),
        obstacle_clearance_m=_require_float(
            m_map, "obstacle_clearance_m", source, "motion", ge=0.0
        ),
        full_body_collision_check=_require_bool(
            m_map, "full_body_collision_check", source, "motion"
        ),
        floor_contact_required=_require_bool(m_map, "floor_contact_required", source, "motion"),
        furniture_contact_allowed=contacts,
    )

    # weighted_actions
    wa_map = _require_mapping(
        _require_key(root, "weighted_actions", source), source, "weighted_actions"
    )
    if len(wa_map) == 0:
        raise _fail(source, "weighted_actions", "must define at least one action weight")
    weighted_actions: dict[str, float] = {}
    for action_name, weight_raw in wa_map.items():
        if not isinstance(action_name, str) or not action_name.strip():
            raise _fail(source, "weighted_actions", "action name must be a non-empty string")
        weighted_actions[action_name] = _as_float(
            weight_raw, source, f"weighted_actions.{action_name}", ge=0.0
        )
    if sum(weighted_actions.values()) <= 0.0:
        raise _fail(source, "weighted_actions", "total action weight sum must be > 0")

    # hard_negative_actions
    raw_hn = _require_list(root, "hard_negative_actions", source, non_empty=True)
    hard_negatives = tuple(
        _require_non_empty_str_item(item, source, f"hard_negative_actions[{idx}]")
        for idx, item in enumerate(raw_hn)
    )

    # fall_model
    fm_map = _require_mapping(_require_key(root, "fall_model", source), source, "fall_model")
    raw_phases = _require_list(fm_map, "phases", source, "fall_model", non_empty=True)
    phases = tuple(
        _require_non_empty_str_item(item, source, f"fall_model.phases[{idx}]")
        for idx, item in enumerate(raw_phases)
    )
    fall_model = FallModelConfig(
        phases=phases,
        use_rigid_body_or_contact_aware_trajectory=_require_bool(
            fm_map, "use_rigid_body_or_contact_aware_trajectory", source, "fall_model"
        ),
        do_not_teleport_pose=_require_bool(fm_map, "do_not_teleport_pose", source, "fall_model"),
        randomize_initial_balance_direction=_require_bool(
            fm_map, "randomize_initial_balance_direction", source, "fall_model"
        ),
        ground_truth_event_timestamp_required=_require_bool(
            fm_map, "ground_truth_event_timestamp_required", source, "fall_model"
        ),
        impact_acceleration_status=_require_str(
            fm_map, "impact_acceleration_status", source, "fall_model"
        ),
    )

    # seeding
    seed_map = _require_mapping(_require_key(root, "seeding", source), source, "seeding")
    raw_streams = _require_list(
        seed_map, "independent_streams", source, "seeding", non_empty=True
    )
    streams: list[str] = []
    seen_streams: set[str] = set()
    for idx, item in enumerate(raw_streams):
        name = _require_non_empty_str_item(item, source, f"seeding.independent_streams[{idx}]")
        if name in seen_streams:
            raise _fail(
                source,
                f"seeding.independent_streams[{idx}]",
                f"duplicate stream name {name!r}",
            )
        seen_streams.add(name)
        streams.append(name)

    seeding_cfg = SeedingConfig(
        global_seed_source=_require_str(seed_map, "global_seed_source", source, "seeding"),
        independent_streams=tuple(streams),
        derivation=_require_str(seed_map, "derivation", source, "seeding"),
    )

    return MotionConfig(
        schema_version=schema_version,
        actor=actor_cfg,
        scenario=scenario_cfg,
        motion=motion_params,
        weighted_actions=weighted_actions,
        hard_negative_actions=hard_negatives,
        fall_model=fall_model,
        seeding=seeding_cfg,
    )


def _require_non_empty_str_item(item: Any, source: str, path: str) -> str:
    if not isinstance(item, str) or not item.strip():
        raise _fail(source, path, f"expected a non-empty string, got {item!r}")
    return item


def load_motion_config(path: str | Path) -> MotionConfig:
    file_path = Path(path)
    raw = _load_yaml_mapping(file_path)
    return parse_motion_config(raw, source=file_path.name)


# ---------------------------------------------------------------------------
# Simulation / UI profile (sim_profile.yaml)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendConfig:
    host: str
    port: int
    websocket_path: str
    frame_rate_hz: float
    default_mode: str
    modes: tuple[str, ...]
    max_cached_frames: int


@dataclass(frozen=True)
class FrontendConfig:
    host: str
    port: int
    framework: str
    four_panel_layout: bool
    shared_frame_clock: bool
    show_skeleton: bool
    show_ground_truth_overlay_on_point_cloud_default: bool


@dataclass(frozen=True)
class ControlsConfig:
    play_pause: bool
    single_step: bool
    seed_selector: bool
    scenario_mode_selector: bool
    speed_values: tuple[float, ...]
    impairment_toggles: bool
    regenerate_scenario: bool


@dataclass(frozen=True)
class PanelConfig:
    id: str
    title: str
    content: str


@dataclass(frozen=True)
class OutputsConfig:
    reports_dir: str
    data_dir: str
    export_format: str
    save_config_snapshot: bool
    save_seed_map: bool


@dataclass(frozen=True)
class SimProfileConfig:
    schema_version: str
    backend: BackendConfig
    frontend: FrontendConfig
    controls: ControlsConfig
    panels: tuple[PanelConfig, ...]
    outputs: OutputsConfig


def parse_sim_profile(
    raw: Mapping[str, Any], *, source: str = "sim_profile.yaml"
) -> SimProfileConfig:
    root = _require_mapping(raw, source, "<root>")
    schema_version = _require_str(root, "schema_version", source)

    # backend
    be_map = _require_mapping(_require_key(root, "backend", source), source, "backend")
    ws_path = _require_str(be_map, "websocket_path", source, "backend")
    if not ws_path.startswith("/"):
        raise _fail(source, "backend.websocket_path", "websocket_path must start with '/'")
    raw_modes = _require_list(be_map, "modes", source, "backend", non_empty=True)
    be_modes = tuple(
        _require_non_empty_str_item(item, source, f"backend.modes[{idx}]")
        for idx, item in enumerate(raw_modes)
    )
    be_default_mode = _require_str(be_map, "default_mode", source, "backend")
    if be_default_mode not in be_modes:
        raise _fail(
            source,
            "backend.default_mode",
            f"default_mode {be_default_mode!r} must be listed in backend.modes ({list(be_modes)})",
        )
    backend_cfg = BackendConfig(
        host=_require_str(be_map, "host", source, "backend"),
        port=_require_int(be_map, "port", source, "backend", min_value=1, max_value=65535),
        websocket_path=ws_path,
        frame_rate_hz=_require_float(be_map, "frame_rate_hz", source, "backend", gt=0.0),
        default_mode=be_default_mode,
        modes=be_modes,
        max_cached_frames=_require_int(be_map, "max_cached_frames", source, "backend", min_value=1),
    )

    # frontend
    fe_map = _require_mapping(_require_key(root, "frontend", source), source, "frontend")
    show_skeleton = _require_bool(fe_map, "show_skeleton", source, "frontend")
    if show_skeleton:
        raise _fail(
            source,
            "frontend.show_skeleton",
            "show_skeleton must be false (internal skeleton must never be rendered as the visible person)",
        )
    four_panel = _require_bool(fe_map, "four_panel_layout", source, "frontend")
    frontend_cfg = FrontendConfig(
        host=_require_str(fe_map, "host", source, "frontend"),
        port=_require_int(fe_map, "port", source, "frontend", min_value=1, max_value=65535),
        framework=_require_str(fe_map, "framework", source, "frontend"),
        four_panel_layout=four_panel,
        shared_frame_clock=_require_bool(fe_map, "shared_frame_clock", source, "frontend"),
        show_skeleton=show_skeleton,
        show_ground_truth_overlay_on_point_cloud_default=_require_bool(
            fe_map,
            "show_ground_truth_overlay_on_point_cloud_default",
            source,
            "frontend",
        ),
    )

    # controls
    ctrl_map = _require_mapping(_require_key(root, "controls", source), source, "controls")
    raw_speeds = _require_list(ctrl_map, "speed_values", source, "controls", non_empty=True)
    speed_values = tuple(
        _as_float(item, source, f"controls.speed_values[{idx}]", gt=0.0)
        for idx, item in enumerate(raw_speeds)
    )
    controls_cfg = ControlsConfig(
        play_pause=_require_bool(ctrl_map, "play_pause", source, "controls"),
        single_step=_require_bool(ctrl_map, "single_step", source, "controls"),
        seed_selector=_require_bool(ctrl_map, "seed_selector", source, "controls"),
        scenario_mode_selector=_require_bool(
            ctrl_map, "scenario_mode_selector", source, "controls"
        ),
        speed_values=speed_values,
        impairment_toggles=_require_bool(ctrl_map, "impairment_toggles", source, "controls"),
        regenerate_scenario=_require_bool(ctrl_map, "regenerate_scenario", source, "controls"),
    )

    # panels
    raw_panels = _require_list(root, "panels", source, non_empty=True)
    panels: list[PanelConfig] = []
    seen_panel_ids: set[str] = set()
    for idx, item in enumerate(raw_panels):
        p_path = f"panels[{idx}]"
        p_map = _require_mapping(item, source, p_path)
        p_id = _require_str(p_map, "id", source, p_path)
        if p_id in seen_panel_ids:
            raise _fail(source, f"{p_path}.id", f"duplicate panel id {p_id!r}")
        seen_panel_ids.add(p_id)
        panels.append(
            PanelConfig(
                id=p_id,
                title=_require_str(p_map, "title", source, p_path),
                content=_require_str(p_map, "content", source, p_path),
            )
        )
    if four_panel and len(panels) != 4:
        raise _fail(
            source,
            "panels",
            f"four_panel_layout is true, so exactly 4 panels must be configured, got {len(panels)}",
        )

    # outputs
    out_map = _require_mapping(_require_key(root, "outputs", source), source, "outputs")
    outputs_cfg = OutputsConfig(
        reports_dir=_require_str(out_map, "reports_dir", source, "outputs"),
        data_dir=_require_str(out_map, "data_dir", source, "outputs"),
        export_format=_require_str(out_map, "export_format", source, "outputs"),
        save_config_snapshot=_require_bool(out_map, "save_config_snapshot", source, "outputs"),
        save_seed_map=_require_bool(out_map, "save_seed_map", source, "outputs"),
    )

    return SimProfileConfig(
        schema_version=schema_version,
        backend=backend_cfg,
        frontend=frontend_cfg,
        controls=controls_cfg,
        panels=tuple(panels),
        outputs=outputs_cfg,
    )


def load_sim_profile(path: str | Path) -> SimProfileConfig:
    file_path = Path(path)
    raw = _load_yaml_mapping(file_path)
    return parse_sim_profile(raw, source=file_path.name)


# ---------------------------------------------------------------------------
# Combined ProjectConfig and cross-file validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectConfig:
    radar: RadarProfile
    expected_derived: ExpectedDerivedMetrics | None
    derived: DerivedRadarMetrics
    scene: SceneConfig
    materials: MaterialDatabase
    motion: MotionConfig
    sim_profile: SimProfileConfig
    config_hash: str


def compute_config_hash(raw_documents: Mapping[str, Any]) -> str:
    """Compute a stable SHA-256 hex digest over canonical JSON config contents."""
    canonical = json.dumps(raw_documents, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_cross_references(
    *,
    radar: RadarProfile,
    scene: SceneConfig,
    materials: MaterialDatabase,
    motion: MotionConfig,
    sim_profile: SimProfileConfig,
    verify_profile_id: bool = True,
    verify_frame_rate: bool = True,
) -> None:
    """Verify cross-file references between scene, materials, radar, motion, and sim profile."""
    missing_materials = sorted(
        mat_id for mat_id in scene.referenced_material_ids() if not materials.has_material(mat_id)
    )
    if missing_materials:
        raise ConfigValidationError(
            f"Scene references unknown material_id(s) not found in material_db: {missing_materials}",
            source="scene.json",
            field_path="material_id",
        )

    if verify_profile_id and scene.radar.profile_id != radar.profile_id:
        raise ConfigValidationError(
            f"scene.radar.profile_id ({scene.radar.profile_id!r}) does not match "
            f"loaded RadarProfile id ({radar.profile_id!r})",
            source="scene.json",
            field_path="radar.profile_id",
        )

    if verify_frame_rate and abs(sim_profile.backend.frame_rate_hz - radar.waveform.frame_rate_hz) > 1e-6:
        raise ConfigValidationError(
            f"sim_profile.backend.frame_rate_hz ({sim_profile.backend.frame_rate_hz}) does not match "
            f"radar_config.waveform.frame_rate_hz ({radar.waveform.frame_rate_hz})",
            source="sim_profile.yaml",
            field_path="backend.frame_rate_hz",
        )

    if scene.initial_actor is not None and scene.initial_actor.actor_id != motion.actor.actor_id:
        raise ConfigValidationError(
            f"scene.initial_actor.actor_id ({scene.initial_actor.actor_id!r}) does not match "
            f"motion_config.actor.actor_id ({motion.actor.actor_id!r})",
            source="scene.json",
            field_path="initial_actor.actor_id",
        )


def load_project_config(
    root_dir: str | Path = ".",
    *,
    radar_path: str | Path | None = None,
    scene_path: str | Path | None = None,
    material_db_path: str | Path | None = None,
    motion_path: str | Path | None = None,
    sim_profile_path: str | Path | None = None,
    require_expected_derived: bool = True,
    validate_expected: bool = True,
    verify_profile_id: bool = True,
    verify_frame_rate: bool = True,
) -> ProjectConfig:
    """Load and strictly validate all five project configuration files."""
    root = Path(root_dir)
    r_path = Path(radar_path) if radar_path is not None else root / "radar_config.yaml"
    s_path = Path(scene_path) if scene_path is not None else root / "scene.json"
    mat_path = Path(material_db_path) if material_db_path is not None else root / "material_db.yaml"
    mot_path = Path(motion_path) if motion_path is not None else root / "motion_config.yaml"
    sim_path = Path(sim_profile_path) if sim_profile_path is not None else root / "sim_profile.yaml"

    raw_radar = _load_yaml_mapping(r_path)
    raw_scene = _load_json_mapping(s_path)
    raw_materials = _load_yaml_mapping(mat_path)
    raw_motion = _load_yaml_mapping(mot_path)
    raw_sim = _load_yaml_mapping(sim_path)

    radar, expected_derived = parse_radar_profile(
        raw_radar,
        source=r_path.name,
        require_expected_derived=require_expected_derived,
        validate_expected=validate_expected,
    )
    derived = compute_derived_metrics(radar)
    scene = parse_scene_config(raw_scene, source=s_path.name)
    materials = parse_material_db(raw_materials, source=mat_path.name)
    motion = parse_motion_config(raw_motion, source=mot_path.name)
    sim_profile = parse_sim_profile(raw_sim, source=sim_path.name)

    validate_cross_references(
        radar=radar,
        scene=scene,
        materials=materials,
        motion=motion,
        sim_profile=sim_profile,
        verify_profile_id=verify_profile_id,
        verify_frame_rate=verify_frame_rate,
    )

    config_hash = compute_config_hash(
        {
            "radar_config": raw_radar,
            "scene": raw_scene,
            "material_db": raw_materials,
            "motion_config": raw_motion,
            "sim_profile": raw_sim,
        }
    )

    return ProjectConfig(
        radar=radar,
        expected_derived=expected_derived,
        derived=derived,
        scene=scene,
        materials=materials,
        motion=motion,
        sim_profile=sim_profile,
        config_hash=config_hash,
    )
