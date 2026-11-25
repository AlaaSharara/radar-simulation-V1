"""Material lookup and complex relative permittivity helpers.

Sign convention (from ``material_db.yaml`` and ``INTERFACES.md``):
    epsilon_r = epsilon_real - j * epsilon_imag
with epsilon_real >= 1.0 and epsilon_imag >= 0.0 for passive media.

When a material defines a conductivity model sigma(f) [S/m], loss is converted
consistently via:
    epsilon_imag(f) = sigma(f) / (2 * pi * f * epsilon_0)
without double-counting any separate dielectric loss term.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from radarsim.config import MaterialDatabase, MaterialEntry

# CODATA 2018 vacuum electric permittivity [F/m]
EPSILON_0_F_PER_M: float = 8.8541878128e-12

_FREQUENCY_UNIT_SCALE_HZ: dict[str, float] = {
    "Hz": 1.0,
    "kHz": 1.0e3,
    "MHz": 1.0e6,
    "GHz": 1.0e9,
}


@dataclass(frozen=True)
class MaterialProperties:
    """Resolved electromagnetic and surface properties for a material at a frequency."""

    requested_id: str
    resolved_id: str
    frequency_hz: float
    model: str
    epsilon_real: float
    epsilon_imag: float
    epsilon_r: complex
    conductivity_s_per_m: float
    is_pec: bool
    roughness_rms_m: float
    thickness_default_m: float | None
    frequency_extrapolated: bool
    provenance: str
    source_url: str | None = None
    uncertainty: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary (no Python complex values)."""
        return {
            "requested_id": self.requested_id,
            "resolved_id": self.resolved_id,
            "frequency_hz": self.frequency_hz,
            "model": self.model,
            "epsilon_real": self.epsilon_real,
            "epsilon_imag": self.epsilon_imag,
            "sign_convention": "epsilon_real - j*epsilon_imag",
            "conductivity_s_per_m": self.conductivity_s_per_m,
            "is_pec": self.is_pec,
            "roughness_rms_m": self.roughness_rms_m,
            "thickness_default_m": self.thickness_default_m,
            "frequency_extrapolated": self.frequency_extrapolated,
            "provenance": self.provenance,
            "source_url": self.source_url,
            "uncertainty": self.uncertainty,
            "notes": self.notes,
        }


def conductivity_s_per_m(entry: MaterialEntry, frequency_hz: float) -> float:
    """Evaluate a material's conductivity model [S/m] at ``frequency_hz``."""
    if frequency_hz <= 0.0 or not math.isfinite(frequency_hz):
        raise ValueError(f"frequency_hz must be positive and finite, got {frequency_hz!r}")
    model = entry.conductivity_model
    if model is None:
        return 0.0
    if model.type != "power_law":
        raise ValueError(
            f"Unsupported conductivity_model.type {model.type!r} for material {entry.material_id!r}"
        )
    if model.frequency_unit not in _FREQUENCY_UNIT_SCALE_HZ:
        raise ValueError(
            f"Unsupported conductivity_model.frequency_unit {model.frequency_unit!r} "
            f"for material {entry.material_id!r}"
        )
    f_scaled = frequency_hz / _FREQUENCY_UNIT_SCALE_HZ[model.frequency_unit]
    return float(model.coefficient * (f_scaled ** model.exponent))


def lookup_material_properties(
    material_db: MaterialDatabase,
    material_id: str,
    frequency_hz: float | None = None,
) -> MaterialProperties:
    """Resolve ``material_id`` (including aliases) and compute complex permittivity."""
    freq = material_db.reference_frequency_hz if frequency_hz is None else float(frequency_hz)
    if freq <= 0.0 or not math.isfinite(freq):
        raise ValueError(f"frequency_hz must be positive and finite, got {freq!r}")

    entry = material_db.resolve(material_id)
    sigma = conductivity_s_per_m(entry, freq)

    if entry.conductivity_model is not None:
        eps_imag = sigma / (2.0 * math.pi * freq * EPSILON_0_F_PER_M)
    else:
        assert entry.epsilon_imag is not None
        eps_imag = float(entry.epsilon_imag)

    eps_real = float(entry.epsilon_real)
    eps_r = complex(eps_real, -eps_imag)

    extrapolated = False
    if entry.valid_frequency_hz is not None:
        f_lo, f_hi = entry.valid_frequency_hz
        extrapolated = not (f_lo <= freq <= f_hi)

    return MaterialProperties(
        requested_id=material_id,
        resolved_id=entry.material_id,
        frequency_hz=freq,
        model=entry.model,
        epsilon_real=eps_real,
        epsilon_imag=eps_imag,
        epsilon_r=eps_r,
        conductivity_s_per_m=sigma,
        is_pec=(entry.model == "perfect_electric_conductor_boundary"),
        roughness_rms_m=entry.roughness_rms_m,
        thickness_default_m=entry.thickness_default_m,
        frequency_extrapolated=extrapolated,
        provenance=entry.provenance,
        source_url=entry.source_url,
        uncertainty=entry.uncertainty,
        notes=entry.notes,
    )


def complex_permittivity(
    material_db: MaterialDatabase,
    material_id: str,
    frequency_hz: float | None = None,
) -> complex:
    """Return complex relative permittivity ``epsilon_real - j * epsilon_imag``."""
    return lookup_material_properties(material_db, material_id, frequency_hz).epsilon_r
