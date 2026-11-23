from pathlib import Path
import importlib.util
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validate_config", ROOT / "scripts" / "validate_config.py")
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_derived_metrics_match_config_expected():
    cfg = yaml.safe_load((ROOT / "radar_config.yaml").read_text(encoding="utf-8"))
    actual = module.derive(cfg)
    expected = cfg["expected_derived"]
    tolerances = expected["tolerances"]
    for key, value in actual.items():
        assert abs(value - float(expected[key])) <= float(tolerances[key]), key


def test_reference_metrics_with_independent_formulas():
    # Recalculate independently here; do not call module.derive().
    cfg = yaml.safe_load((ROOT / "radar_config.yaml").read_text(encoding="utf-8"))
    c = float(cfg["rf"]["speed_of_light_m_per_s"])
    fc = float(cfg["rf"]["carrier_frequency_hz"])
    slope = float(cfg["waveform"]["slope_hz_per_s"])
    sample_rate = float(cfg["waveform"]["adc_sample_rate_hz"])
    adc_count = int(cfg["waveform"]["adc_samples_per_chirp"])
    chirp_period = float(cfg["waveform"]["chirp_period_s"])
    tx_count = int(cfg["rf"]["tx_count"])
    doppler_loops = int(cfg["waveform"]["chirps_per_tx_per_frame"])
    if_max = float(cfg["rf"]["receiver"]["max_if_frequency_hz"])

    wavelength = c / fc
    effective_bw = slope * adc_count / sample_rate
    t_same = chirp_period * tx_count
    independently_calculated = {
        "effective_bandwidth_hz": effective_bw,
        "range_resolution_m": c / (2.0 * effective_bw),
        "max_unambiguous_range_m": c * if_max / (2.0 * slope),
        "wavelength_m": wavelength,
        "same_tx_chirp_interval_s": t_same,
        "max_unambiguous_radial_velocity_mps": wavelength / (4.0 * t_same),
        "velocity_resolution_mps": wavelength / (2.0 * doppler_loops * t_same),
    }
    expected = cfg["expected_derived"]
    tolerances = expected["tolerances"]
    for key, value in independently_calculated.items():
        assert abs(value - float(expected[key])) <= float(tolerances[key]), key


def test_tdm_same_tx_interval_is_three_chirp_periods():
    cfg = yaml.safe_load((ROOT / "radar_config.yaml").read_text(encoding="utf-8"))
    same_tx = float(cfg["waveform"]["chirp_period_s"]) * int(cfg["rf"]["tx_count"])
    assert abs(same_tx - 180e-6) < 1e-12
