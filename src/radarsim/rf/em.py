"""Complex Fresnel TE/TM reflection and 2x2 transfer-matrix multilayer slab calculations.

Conventions (consistent with ``material_db.yaml`` and ``radarsim.scene.materials``):
- Time-harmonic factor: ``exp(+j * omega * t)``; forward wave propagates as
  ``exp(-j * k * r)`` with ``Im(k) <= 0`` so passive lossy media attenuate.
- Complex relative permittivity: ``epsilon_r = epsilon_real - j * epsilon_imag``
  with ``epsilon_real >= 1.0`` and ``epsilon_imag >= 0.0``.
- Non-magnetic media: ``mu_r = 1.0``.
- Incidence angle ``theta_i`` is the angle between the incident ray and the
  surface normal, in ``[0, pi/2]``.
- Polarizations:
  - ``TE`` (s-polarization): electric field perpendicular to plane of incidence.
  - ``TM`` (p-polarization): electric field parallel to plane of incidence.
    Note on sign convention: ``fresnel_single_interface`` uses standard
    plane-of-incidence p-polarization (``+1`` at PEC), whereas
    ``multilayer_reflection_transmission`` uses Born & Wolf tangential ``E`` /
    ``H`` wave-admittance matrices (``Y_TE = n_z``, ``Y_TM = eps / n_z``) so
    ``r_TM_tmm = -r_TM_fresnel_p`` (identical ``|R|`` and Brewster angle, and
    continuous at ``theta = 0`` and ``d -> 0`` onto PEC where tangential ``E``
    ratio is ``-1`` for both TE and TM).
- Dielectric/conductivity losses enter exactly once through ``epsilon_r`` from
  ``radarsim.scene.materials.lookup_material_properties``.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Sequence

from radarsim.config import MaterialDatabase
from radarsim.scene import SurfaceLayer, lookup_material_properties


def _passive_sqrt(z: complex) -> complex:
    """Principal complex square root choosing the passive propagating branch ``Im(w) <= 0``.

    Under ``exp(+j*omega*t - j*k*z)``, ``Im(k_z) <= 0`` guarantees exponential
    decay (never exponential growth) inside lossy media, while ``Re(k_z) >= 0``
    selects forward propagation.
    """
    w = cmath.sqrt(z)
    if w.imag > 0.0 or (w.imag == 0.0 and w.real < 0.0):
        w = -w
    return w


def fresnel_single_interface(
    epsilon_r_2: complex,
    theta_i_rad: float,
    *,
    polarization: str = "TE",
    epsilon_r_1: complex = 1.0 + 0.0j,
    is_pec: bool = False,
) -> complex:
    """Complex Fresnel reflection coefficient for a single planar interface.

    Parameters
    ----------
    epsilon_r_2:
        Complex relative permittivity ``eps_real - j*eps_imag`` of medium 2.
    theta_i_rad:
        Incidence angle in ``[0, pi/2]`` between ray and surface normal.
    polarization:
        ``"TE"`` (s-polarized) or ``"TM"`` (p-polarized).
    epsilon_r_1:
        Complex relative permittivity of incident medium (default air = 1.0).
    is_pec:
        If True, returns the ideal PEC boundary reflection (``-1`` TE, ``+1`` TM).
    """
    if polarization not in ("TE", "TM"):
        raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}")
    theta = max(0.0, min(0.5 * math.pi, float(theta_i_rad)))
    if is_pec:
        return (-1.0 + 0.0j) if polarization == "TE" else (1.0 + 0.0j)

    sin2 = math.sin(theta) ** 2
    nz1 = _passive_sqrt(epsilon_r_1 - sin2)
    nz2 = _passive_sqrt(epsilon_r_2 - sin2)

    if polarization == "TE":
        denom = nz1 + nz2
        if abs(denom) < 1e-18:
            return -1.0 + 0.0j
        return (nz1 - nz2) / denom

    # TM polarization
    denom = epsilon_r_2 * nz1 + epsilon_r_1 * nz2
    if abs(denom) < 1e-18:
        return 1.0 + 0.0j
    return (epsilon_r_2 * nz1 - epsilon_r_1 * nz2) / denom


def _effective_admittance(epsilon_r: complex, sin2_theta: float, polarization: str) -> complex:
    nz = _passive_sqrt(epsilon_r - sin2_theta)
    if polarization == "TE":
        return nz
    # TM effective admittance: epsilon_r / nz
    if abs(nz) < 1e-15:
        return complex(1e15, 0.0)
    return epsilon_r / nz


def multilayer_reflection_transmission(
    layer_epsilons: Sequence[complex],
    layer_thicknesses_m: Sequence[float],
    theta_i_rad: float,
    frequency_hz: float,
    *,
    polarization: str = "TE",
    speed_of_light_m_per_s: float = 299792458.0,
    epsilon_in: complex = 1.0 + 0.0j,
    epsilon_out: complex = 1.0 + 0.0j,
    layer_is_pec: Sequence[bool] | None = None,
) -> tuple[complex, complex]:
    """Exact 2x2 characteristic transfer-matrix reflection ``R`` and transmission ``T``.

    Computes coherent multiple internal reflections and dielectric/conductivity
    absorption across an arbitrary stack of homogeneous planar layers (such as
    the configured plasterboard / air-cavity / plasterboard wall). Loss is
    applied once inside each layer's complex wavenumber ``k_z``.

    For optically opaque / PEC layers (``-Im(k_z * d) > 25``), falls back safely
    to the front-facing stack without floating-point overflow (``T = 0``).
    """
    if polarization not in ("TE", "TM"):
        raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}")
    if len(layer_epsilons) != len(layer_thicknesses_m):
        raise ValueError("layer_epsilons and layer_thicknesses_m must have equal length")
    if frequency_hz <= 0.0:
        raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")

    theta = max(0.0, min(0.5 * math.pi, float(theta_i_rad)))
    sin2 = math.sin(theta) ** 2
    k0 = 2.0 * math.pi * frequency_hz / speed_of_light_m_per_s
    pec_flags = (
        tuple(layer_is_pec) if layer_is_pec is not None else (False,) * len(layer_epsilons)
    )

    if len(layer_epsilons) == 0:
        r0 = fresnel_single_interface(
            epsilon_out, theta, polarization=polarization, epsilon_r_1=epsilon_in
        )
        return r0, 1.0 + r0

    eta0 = _effective_admittance(epsilon_in, sin2, polarization)
    eta_out = _effective_admittance(epsilon_out, sin2, polarization)

    # Accumulate 2x2 characteristic matrix M = M_1 @ M_2 @ ... @ M_N
    m00, m01 = 1.0 + 0.0j, 0.0 + 0.0j
    m10, m11 = 0.0 + 0.0j, 1.0 + 0.0j

    for eps_m, d_m, is_pec in zip(layer_epsilons, layer_thicknesses_m, pec_flags):
        if is_pec:
            # Opaque PEC boundary (E_tan = 0 -> eta_out -> inf for both TE and TM tangential E/H):
            # When preceding stack is empty (M = I, m01 = 0, m11 = 1), limit gives r_pec = -1.0,
            # continuous with lim_{d->0} (eta0 * m01 - m11) / (eta0 * m01 + m11) = -1.0.
            if m00 == 1.0 and m01 == 0.0 and m10 == 0.0 and m11 == 1.0:
                return (-1.0 + 0.0j), (0.0 + 0.0j)
            r_pec = (eta0 * m01 - m11) / (eta0 * m01 + m11)
            return r_pec, 0.0 + 0.0j

        if d_m < 0.0:
            raise ValueError(f"Layer thickness must be >= 0, got {d_m}")

        nz_m = _passive_sqrt(eps_m - sin2)
        phi = k0 * nz_m * d_m  # Im(phi) <= 0 for passive lossy media

        # Guard against cosh/sinh overflow in optically opaque thick/lossy layers (e.g. metal or thick concrete)
        if -phi.imag > 25.0:
            eta_semi = _effective_admittance(eps_m, sin2, polarization)
            denom = eta0 * m00 + eta0 * eta_semi * m01 + m10 + eta_semi * m11
            numer = eta0 * m00 + eta0 * eta_semi * m01 - m10 - eta_semi * m11
            r_semi = numer / denom if abs(denom) > 1e-18 else (-1.0 + 0.0j)
            return r_semi, 0.0 + 0.0j

        eta_m = _effective_admittance(eps_m, sin2, polarization)
        c_phi = cmath.cos(phi)
        s_phi = cmath.sin(phi)
        l00 = c_phi
        l01 = (1j / eta_m) * s_phi
        l10 = (1j * eta_m) * s_phi
        l11 = c_phi

        # M = M @ L
        n00 = m00 * l00 + m01 * l10
        n01 = m00 * l01 + m01 * l11
        n10 = m10 * l00 + m11 * l10
        n11 = m10 * l01 + m11 * l11
        m00, m01, m10, m11 = n00, n01, n10, n11

    denom = eta0 * m00 + eta0 * eta_out * m01 + m10 + eta_out * m11
    if abs(denom) < 1e-18:
        return (-1.0 + 0.0j), (0.0 + 0.0j)
    r_tot = (eta0 * m00 + eta0 * eta_out * m01 - m10 - eta_out * m11) / denom
    t_tot = (2.0 * eta0) / denom
    return r_tot, t_tot


def rayleigh_roughness_factor(
    roughness_rms_m: float,
    theta_i_rad: float,
    wavelength_m: float,
) -> float:
    """Coherent specular reflection attenuation due to surface roughness (Kirchhoff/Rayleigh).

    ``rho_s = exp(-2 * (2*pi * sigma_h * cos(theta_i) / lambda)^2)`` in ``(0, 1]``.
    """
    if roughness_rms_m <= 0.0:
        return 1.0
    g = 2.0 * math.pi * roughness_rms_m * max(0.0, math.cos(theta_i_rad)) / wavelength_m
    return float(math.exp(-2.0 * g * g))


@dataclass(frozen=True)
class SurfaceEMResponse:
    """Evaluated reflection and transmission coefficients for a ``SurfacePatch``."""

    reflection_coherent: complex
    reflection_smooth: complex
    transmission: complex
    roughness_factor: float
    diffuse_energy_fraction: float
    method: str
    frequency_extrapolated: bool
    provenance_notes: tuple[str, ...]


def evaluate_surface_em(
    layers: Sequence[SurfaceLayer],
    material_db: MaterialDatabase,
    theta_i_rad: float,
    frequency_hz: float,
    *,
    polarization: str = "TE",
    speed_of_light_m_per_s: float = 299792458.0,
) -> SurfaceEMResponse:
    """Compute reflection and transmission for a configured single- or multi-layer surface."""
    if not layers:
        return SurfaceEMResponse(
            reflection_coherent=0.0 + 0.0j,
            reflection_smooth=0.0 + 0.0j,
            transmission=1.0 + 0.0j,
            roughness_factor=1.0,
            diffuse_energy_fraction=0.0,
            method="empty_air_interface",
            frequency_extrapolated=False,
            provenance_notes=(),
        )

    eps_list: list[complex] = []
    thick_list: list[float] = []
    pec_list: list[bool] = []
    extrapolated = False
    notes: list[str] = []

    front_props = lookup_material_properties(material_db, layers[0].material_id, frequency_hz)
    for layer in layers:
        props = lookup_material_properties(material_db, layer.material_id, frequency_hz)
        eps_list.append(props.epsilon_r)
        thick_list.append(layer.thickness_m)
        pec_list.append(props.is_pec)
        if props.frequency_extrapolated:
            extrapolated = True
        notes.append(f"{layer.material_id}:{props.provenance}")

    r_smooth, t_slab = multilayer_reflection_transmission(
        eps_list,
        thick_list,
        theta_i_rad,
        frequency_hz,
        polarization=polarization,
        speed_of_light_m_per_s=speed_of_light_m_per_s,
        layer_is_pec=pec_list,
    )

    wavelength_m = speed_of_light_m_per_s / frequency_hz
    rho_s = rayleigh_roughness_factor(front_props.roughness_rms_m, theta_i_rad, wavelength_m)
    r_coh = rho_s * r_smooth
    # Remaining passive scattered fraction available for stochastic diffuse model
    diffuse_frac = max(0.0, 1.0 - rho_s * rho_s)

    method = (
        "transfer_matrix_multilayer_stack"
        if len(layers) > 1
        else "transfer_matrix_finite_slab"
    )
    return SurfaceEMResponse(
        reflection_coherent=r_coh,
        reflection_smooth=r_smooth,
        transmission=t_slab,
        roughness_factor=rho_s,
        diffuse_energy_fraction=diffuse_frac,
        method=method,
        frequency_extrapolated=extrapolated,
        provenance_notes=tuple(notes),
    )
