"""Furniture RF sanity report: per-panel reflection/transmission table + multi-angle picture.

Usage (from the repo root):
    set PYTHONPATH=src            (PowerShell: $env:PYTHONPATH="src")
    python scripts/furniture_rf_check.py [--scene scene.json] [--material-db material_db.yaml] [--out reports]

Outputs:
    reports/furniture_rf_check.png   4 angles (iso / top / radar-wall elevation / side), panels coloured by |R| at 45 deg
    reports/furniture_rf_check.md    per-panel table (layers, |R| at 0/45/70 deg, |T| at 45 deg, roughness factor)
It uses exactly the transfer-matrix code of the RF engine, so the numbers are what the simulation uses.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from radarsim.rf import evaluate_surface_em  # noqa: E402
from radarsim.scene import load_scene  # noqa: E402

F_HZ = 61.0e9


def _db(x: complex) -> float:
    return 20.0 * math.log10(max(abs(x), 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene.json")
    ap.add_argument("--material-db", default="material_db.yaml")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = load_scene(a.scene, a.material_db)

    rows = []
    polys, colors = [], []
    for obj in scene.objects:
        for sf in obj.surfaces:
            r = {d: _db(evaluate_surface_em(sf.layers, scene.materials, math.radians(d), F_HZ).reflection_coherent) for d in (0, 45, 70)}
            em45 = evaluate_surface_em(sf.layers, scene.materials, math.radians(45), F_HZ)
            rows.append((obj.object_id, sf.surface_id, sf.layers, r, _db(em45.transmission), em45.roughness_factor, sf.area_m2))
            polys.append(np.asarray(sf.vertices_m))
            colors.append(r[45])

    # ---- markdown table -------------------------------------------------
    lines = [
        "# Furniture RF check",
        "",
        f"Scene `{scene.scene_id}`, carrier {F_HZ/1e9:.1f} GHz, TE polarisation. |R| is the *coherent* (roughness-reduced) reflection;",
        "the missing energy is returned by the diffuse-scatter model. Values come from the same TMM code as the RF engine.",
        "",
        "| object | panel | layers (material, mm) | area m2 | R 0deg dB | R 45deg dB | R 70deg dB | T 45deg dB | rough. factor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for oid, sid, layers, r, t, rho, area in rows:
        ls = " + ".join(f"{l.material_id} {l.thickness_m*1000:.0f}" for l in layers)
        lines.append(f"| {oid} | {sid} | {ls} | {area:.2f} | {r[0]:.1f} | {r[45]:.1f} | {r[70]:.1f} | {t:.1f} | {rho:.2f} |")
    (out / "furniture_rf_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- picture ---------------------------------------------------------
    cmap = plt.get_cmap("turbo")
    vmin, vmax = -40.0, -4.0
    norm = lambda v: cmap((min(max(v, vmin), vmax) - vmin) / (vmax - vmin))  # noqa: E731
    face_cols = [norm(c) for c in colors]
    W, D, H = scene.width_x_m, scene.depth_y_m, scene.height_z_m
    rp = np.asarray(scene.radar_frame.position_m)

    fig = plt.figure(figsize=(14, 10), dpi=110)
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.add_collection3d(Poly3DCollection(polys, facecolors=face_cols, edgecolors="k", linewidths=0.3, alpha=0.9))
    ax.set_xlim(0, W); ax.set_ylim(0, D); ax.set_zlim(0, H)
    ax.set_box_aspect((W, D, H)); ax.view_init(elev=28, azim=-60)
    ax.scatter(*rp, c="red", s=40); ax.set_title("Iso (radar = red)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    def proj(axn, ia, ib, title, la, lb, xlim, ylim, flip=False):
        a2 = fig.add_subplot(2, 2, axn)
        verts = [p[:, [ia, ib]] for p in polys]
        a2.add_collection(PolyCollection(verts, facecolors=face_cols, edgecolors="k", linewidths=0.4, alpha=0.75))
        a2.plot([0, xlim, xlim, 0, 0], [0, 0, ylim, ylim, 0], "k-", lw=0.8)
        a2.scatter([rp[ia]], [rp[ib]], c="red", s=30, zorder=5)
        a2.set_xlim(-0.1, xlim + 0.1); a2.set_ylim(-0.1, ylim + 0.1); a2.set_aspect("equal")
        a2.set_title(title); a2.set_xlabel(la); a2.set_ylabel(lb)
        return a2

    proj(2, 0, 1, "Top (x-y); radar wall at y=0", "x [m]", "y [m]", W, D)
    proj(3, 0, 2, "Elevation from the radar wall (x-z)", "x [m]", "z [m]", W, H)
    a4 = proj(4, 1, 2, "Side elevation (y-z)", "y [m]", "z [m]", D, H)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, label="coherent |R| at 45 deg incidence [dB]")
    fig.suptitle("Furniture RF panels as the simulator sees them (all panels, coloured by reflectivity)")
    fig.savefig(out / "furniture_rf_check.png", bbox_inches="tight")
    print(f"wrote {out/'furniture_rf_check.png'} and {out/'furniture_rf_check.md'} ({len(rows)} panels)")


if __name__ == "__main__":
    main()
