"""Detailed, physically motivated RF panel models for furniture.

Why this exists
---------------
The first build treated every furniture object as a *solid six-face box* with one
slab per face (wood = 50 mm, fabric = 10 mm).  That is wrong in several ways at 60 GHz
(wavelength ~ 4.9 mm):

* a table or chair is mostly *air* (a top plate on thin legs), but the box blocked and
  reflected rays like a solid wooden block, hiding the legs of a person standing behind it;
* a wardrobe is a *hollow* carcass of ~18 mm board, not a 50 mm solid slab;
* the sofa / mattress were a single 10 mm fabric sheet (eps 1.5).  A 10 mm slab of that
  material is almost exactly an anti-reflection layer at 61 GHz (|R| ~ -56 dB at normal
  incidence): the sofa and bed were practically invisible to the radar;
* the 'bottom' face of every box lay on the floor and produced a duplicate reflector.

The models below are built from the SAME dimensions as ``scene.json`` and the same part
layout as the browser's visible furniture (``viz/web/src/lib/sceneBuilder.js``), so what the
radar "sees" matches what the Scene panel draws.  Each part is a planar rectangle carrying a
layer stack that is evaluated with the existing transfer-matrix code (``rf/em.py``).

Known limitations (documented, intentional)
-------------------------------------------
* The engine uses image-method specular reflection from finite rectangles.  Features
  smaller than the first Fresnel zone (~12 cm at 3 m range) such as legs, handles, hinges
  and individual chair slats cannot be represented faithfully and are omitted.
* Foam permittivity is an assumption (see ``material_db.yaml``).  Inner-spring mattress coils
  (steel) are not modelled.
* Unknown ``kind`` values fall back to the legacy solid-box model.

Local frame: ``lx`` along the object width (box axis 0), ``ly`` along the depth (axis 1,
``+ly`` is the back / headboard side, towards the back wall for yaw 0), ``lh`` height above
the floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from radarsim.scene.geometry import OrientedBox, SurfaceLayer, SurfacePatch

Layers = Sequence[tuple[str, float]]

SUPPORTED_KINDS: tuple[str, ...] = ("bed", "sofa", "table", "chair", "wardrobe")

FABRIC_COVER_M = 0.002  # thin upholstery cover


@dataclass(frozen=True)
class _Frame:
    """Maps object-local (lx, ly, lh) to world coordinates."""

    origin_xy: tuple[float, float]
    z_floor: float
    u_x: np.ndarray  # world direction of local +x
    u_y: np.ndarray  # world direction of local +y

    def point(self, lx: float, ly: float, lh: float) -> np.ndarray:
        return np.array(
            [
                self.origin_xy[0] + lx * self.u_x[0] + ly * self.u_y[0],
                self.origin_xy[1] + lx * self.u_x[1] + ly * self.u_y[1],
                self.z_floor + lh,
            ],
            dtype=np.float64,
        )

    def vec(self, dx: float, dy: float, dh: float) -> np.ndarray:
        return np.array(
            [
                dx * self.u_x[0] + dy * self.u_y[0],
                dx * self.u_x[1] + dy * self.u_y[1],
                dh,
            ],
            dtype=np.float64,
        )


def _frame_from_box(box: OrientedBox) -> _Frame:
    axes2d = box.axes_2d()
    cx, cy, cz = box.center_m
    hz = box.half_extents_m[2]
    return _Frame(
        origin_xy=(float(cx), float(cy)),
        z_floor=float(cz - hz),
        u_x=np.array([axes2d[0, 0], axes2d[0, 1], 0.0], dtype=np.float64),
        u_y=np.array([axes2d[1, 0], axes2d[1, 1], 0.0], dtype=np.float64),
    )


class _PanelBuilder:
    def __init__(self, obj_id: str, kind: str, frame: _Frame, material_db) -> None:
        self.obj_id = obj_id
        self.kind = kind
        self.frame = frame
        self.db = material_db
        self.patches: list[SurfacePatch] = []

    def _layers(self, spec: Layers) -> tuple[SurfaceLayer, ...]:
        out = []
        for mat_id, t in spec:
            self.db.resolve(mat_id)  # fail early on unknown material ids
            out.append(SurfaceLayer(material_id=mat_id, thickness_m=float(t)))
        return tuple(out)

    def quad(
        self,
        part: str,
        face: str,
        origin: tuple[float, float, float],
        u: tuple[float, float, float],
        v: tuple[float, float, float],
        spec: Layers,
    ) -> None:
        layers = self._layers(spec)
        self.patches.append(
            SurfacePatch.from_quad(
                surface_id=f"{self.obj_id}_{part}_{face}",
                parent_id=self.obj_id,
                kind="furniture",
                origin_m=self.frame.point(*origin),
                u_vec_m=self.frame.vec(*u),
                v_vec_m=self.frame.vec(*v),
                material_id=layers[0].material_id,
                layers=layers,
                is_passable=False,
                metadata={
                    "object_kind": self.kind,
                    "part": part,
                    "face": face,
                    "rf_model": "detailed_panels_v2",
                },
            )
        )

    def box(
        self,
        part: str,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        h0: float,
        h1: float,
        faces: Mapping[str, Layers],
    ) -> None:
        """Add the requested outward-facing faces of an axis-aligned (local) box."""
        w, d, h = x1 - x0, y1 - y0, h1 - h0
        if w <= 1e-6 or d <= 1e-6 or h <= 1e-6:
            return
        specs = {
            "top": ((x0, y0, h1), (w, 0, 0), (0, d, 0)),
            "bottom": ((x0, y1, h0), (w, 0, 0), (0, -d, 0)),
            "front": ((x0, y0, h0), (w, 0, 0), (0, 0, h)),
            "back": ((x1, y1, h0), (-w, 0, 0), (0, 0, h)),
            "left": ((x0, y1, h0), (0, -d, 0), (0, 0, h)),
            "right": ((x1, y0, h0), (0, d, 0), (0, 0, h)),
        }
        for face, layer_spec in faces.items():
            o, u, v = specs[face]
            self.quad(part, face, o, u, v, layer_spec)

    def plate(
        self,
        part: str,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        h_top: float,
        spec: Layers,
    ) -> None:
        """Single horizontal plate whose upward-facing surface is at ``h_top``."""
        self.quad(part, "top", (x0, y0, h_top), (x1 - x0, 0, 0), (0, y1 - y0, 0), spec)


# ---------------------------------------------------------------------------
# Per-kind builders
# ---------------------------------------------------------------------------


def _bed(b: _PanelBuilder, dx: float, dy: float, dz: float) -> None:
    hx, hy = dx / 2.0, dy / 2.0
    leg_h = min(0.10, dz * 0.18)
    frame_h = min(0.16, dz * 0.28)
    head_t = min(0.08, dy * 0.06)
    z_frame1 = leg_h + frame_h
    z_matt1 = max(z_frame1 + 0.14, dz - 0.05)  # bedding/pillow allowance (also used by the 3D view)
    matt_h = z_matt1 - z_frame1

    rail = (("wood", 0.025),)
    # Platform rails (3 sides; the 4th is the headboard).  Open underneath: legs only.
    y_end = hy - head_t
    b.box("rail", -hx, hx, -hy, y_end, leg_h, z_frame1, {"left": rail, "right": rail, "front": rail})
    # Padded headboard (solid, full height)
    head = (("wood", 0.03),)
    b.box("headboard", -hx, hx, y_end, hy, leg_h, dz, {"front": head, "back": head, "top": head, "left": head, "right": head})
    # Mattress: fabric cover / foam / wooden slat deck underneath
    m_w = dx - 0.06
    m_d = dy - head_t - 0.04
    yc = -head_t / 2.0
    m0, m1 = yc - m_d / 2.0, yc + m_d / 2.0
    deck_t = 0.02
    top_stack = (("fabric", FABRIC_COVER_M), ("foam", matt_h - FABRIC_COVER_M - deck_t), ("wood", deck_t))
    side_stack = (("fabric", FABRIC_COVER_M), ("foam", 0.25))
    b.box(
        "mattress",
        -m_w / 2.0,
        m_w / 2.0,
        m0,
        m1,
        z_frame1,
        z_matt1,
        {"top": top_stack, "front": side_stack, "left": side_stack, "right": side_stack, "back": side_stack},
    )


def _sofa(b: _PanelBuilder, dx: float, dy: float, dz: float) -> None:
    hx, hy = dx / 2.0, dy / 2.0
    leg_h = min(0.08, dz * 0.10)
    base_h = dz * 0.22
    seat_h = dz * 0.18
    arm_w = min(0.16, dx * 0.12)
    back_t = min(0.20, dy * 0.26)
    arm_h = dz * 0.42
    z_base1 = leg_h + base_h
    z_seat1 = z_base1 + seat_h
    z_arm1 = z_base1 + arm_h

    side = (("fabric", FABRIC_COVER_M), ("foam", 0.10))
    seat_top = (("fabric", FABRIC_COVER_M), ("foam", seat_h - FABRIC_COVER_M), ("chipboard", 0.018))
    # Seat block (base + cushions) between the arms; open underneath (legs)
    b.box("seat", -hx + arm_w, hx - arm_w, -hy, hy, leg_h, z_seat1, {"top": seat_top, "front": side, "back": side})
    # Armrests
    b.box("arm_l", -hx, -hx + arm_w, -hy, hy, leg_h, z_arm1, {"top": side, "front": side, "back": side, "left": side, "right": side})
    b.box("arm_r", hx - arm_w, hx, -hy, hy, leg_h, z_arm1, {"top": side, "front": side, "back": side, "left": side, "right": side})
    # Backrest above the seat, between the arms
    b.box("backrest", -hx + arm_w, hx - arm_w, hy - back_t, hy, z_seat1, dz, {"top": side, "front": side, "back": side, "left": side, "right": side})


def _table(b: _PanelBuilder, dx: float, dy: float, dz: float) -> None:
    top_t = min(0.045, dz * 0.12)
    # A table is a top plate on thin legs: radar can see (and ray-trace) underneath it.
    b.plate("top", -dx / 2.0, dx / 2.0, -dy / 2.0, dy / 2.0, dz, (("wood", top_t),))


def _chair(b: _PanelBuilder, dx: float, dy: float, dz: float) -> None:
    seat_center = dz * 0.46
    seat_t, pad_t = 0.045, 0.035
    h_top = seat_center + seat_t / 2.0 + pad_t
    hx, hy = (dx - 0.02) / 2.0, (dy - 0.02) / 2.0
    b.plate(
        "seat",
        -hx,
        hx,
        -hy,
        hy,
        h_top,
        (("fabric", FABRIC_COVER_M), ("foam", pad_t - FABRIC_COVER_M), ("wood", seat_t)),
    )
    # Slatted backrest at +ly, modelled as one effective wooden panel spanning the slats
    leg_w = 0.038
    post_h = dz * 0.54
    h_lo = seat_center + post_h * 0.35 - 0.03
    h_hi = seat_center + post_h * 0.90 + 0.03
    y_plane = dy / 2.0 - leg_w - 0.0125
    b.quad("backrest", "front", (-(dx - 0.04) / 2.0, y_plane, h_lo), (dx - 0.04, 0, 0), (0, 0, h_hi - h_lo), (("wood", 0.025),))


def _wardrobe(b: _PanelBuilder, dx: float, dy: float, dz: float) -> None:
    hx, hy = dx / 2.0, dy / 2.0
    plinth = 0.07
    board = (("chipboard", 0.018),)
    back_board = (("chipboard", 0.008),)
    # Hollow carcass: 18 mm board, thin back panel, interior is air (rays can enter via the doors/sides)
    b.box(
        "carcass",
        -hx,
        hx,
        -hy,
        hy,
        plinth,
        dz,
        {"top": board, "bottom": board, "front": board, "back": back_board, "left": board, "right": board},
    )


_BUILDERS = {
    "bed": _bed,
    "sofa": _sofa,
    "table": _table,
    "chair": _chair,
    "wardrobe": _wardrobe,
}


def build_detailed_furniture_surfaces(
    obj_id: str,
    kind: str,
    box: OrientedBox,
    material_db,
) -> tuple[SurfacePatch, ...] | None:
    """Return realistic RF panels for ``kind`` or ``None`` when the kind has no detailed model."""
    builder_fn = _BUILDERS.get(kind)
    if builder_fn is None:
        return None
    dx, dy, dz = box.dimensions_m
    b = _PanelBuilder(obj_id, kind, _frame_from_box(box), material_db)
    builder_fn(b, float(dx), float(dy), float(dz))
    return tuple(b.patches)
