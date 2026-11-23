"""Procedural rigged & skinned adult male-presenting human avatar (glTF 2.0 GLB).

Generates a complete, browser-ready binary glTF 2.0 (``.glb``) mesh with a
hidden 17-joint skinning rig matching ``INTERFACES.md``, clothed human
surfaces (skin, hair, casual shirt, trousers, shoes), and a companion JSON
rig mapping for Three.js consumption in T6.

Never renders skeleton bones, stick figures, capsules, or joint markers as the
visible person.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Any, Sequence

import numpy as np

from radarsim.motion.types import JOINT_NAMES, JOINT_PARENTS, Vec3

AVATAR_ASSET_ID: str = "adult_male_default"
NOMINAL_HEIGHT_M: float = 1.78

# Right-handed axis conversion from World Z-up (+X right, +Y forward, +Z up)
# to standard glTF/Three.js Y-up (+X right, +Y up, -Z forward). det = +1.
WORLD_TO_GLTF_MATRIX: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
)


def world_to_gltf(pt_world: Sequence[float]) -> Vec3:
    return (float(pt_world[0]), float(pt_world[2]), -float(pt_world[1]))


def rest_joints_world(height_m: float = NOMINAL_HEIGHT_M, body_scale: float = 1.0) -> dict[str, Vec3]:
    """Compute anatomical neutral standing bind-pose joint coordinates in world Z-up [m].

    Feet rest on ``z = 0``, actor faces ``+Y`` (into room), ``+X`` is actor's right.
    """
    h = float(height_m)
    w = float(body_scale)
    hip_x = 0.095 * (h / 1.78) * w
    sh_x = 0.195 * (h / 1.78) * w
    el_x = 0.225 * (h / 1.78) * w
    wr_x = 0.240 * (h / 1.78) * w
    return {
        "pelvis": (0.0, 0.0, 0.530 * h),
        "spine": (0.0, 0.0, 0.635 * h),
        "chest": (0.0, 0.0, 0.775 * h),
        "neck": (0.0, 0.0, 0.845 * h),
        "head": (0.0, 0.01 * h, 0.925 * h),
        "left_shoulder": (-sh_x, 0.0, 0.805 * h),
        "left_elbow": (-el_x, 0.0, 0.630 * h),
        "left_wrist": (-wr_x, 0.015 * h, 0.475 * h),
        "right_shoulder": (sh_x, 0.0, 0.805 * h),
        "right_elbow": (el_x, 0.0, 0.630 * h),
        "right_wrist": (wr_x, 0.015 * h, 0.475 * h),
        "left_hip": (-hip_x, 0.0, 0.515 * h),
        "left_knee": (-hip_x, 0.008 * h, 0.285 * h),
        "left_ankle": (-hip_x, -0.01 * h, 0.055 * h),
        "right_hip": (hip_x, 0.0, 0.515 * h),
        "right_knee": (hip_x, 0.008 * h, 0.285 * h),
        "right_ankle": (hip_x, -0.01 * h, 0.055 * h),
    }


@dataclass
class _SubMesh:
    material_idx: int
    positions: list[np.ndarray]  # world Z-up (N, 3)
    normals: list[np.ndarray]    # world Z-up (N, 3)
    joints: list[np.ndarray]     # (N, 4) uint16
    weights: list[np.ndarray]    # (N, 4) float32
    indices: list[np.ndarray]    # (M,) uint32
    vertex_offset: int = 0

    def add(
        self,
        pos: np.ndarray,
        norm: np.ndarray,
        jnt: np.ndarray,
        wgt: np.ndarray,
        idx: np.ndarray,
    ) -> None:
        self.positions.append(pos.astype(np.float32))
        self.normals.append(norm.astype(np.float32))
        self.joints.append(jnt.astype(np.uint16))
        self.weights.append(wgt.astype(np.float32))
        self.indices.append((idx + self.vertex_offset).astype(np.uint32))
        self.vertex_offset += int(pos.shape[0])


def _loft_tube(
    rings_centers: Sequence[Vec3],
    rings_radii: Sequence[tuple[float, float]],
    rings_joints: Sequence[tuple[tuple[int, float], ...]],
    *,
    sides: int = 16,
    cap_bottom: bool = True,
    cap_top: bool = True,
    axis: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth closed lofted elliptical surface with per-ring bone weights."""
    n_rings = len(rings_centers)
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    verts: list[np.ndarray] = []
    jnts: list[np.ndarray] = []
    wgts: list[np.ndarray] = []

    def _encode_jw(pairs: Sequence[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        j_arr = np.zeros(4, dtype=np.uint16)
        w_arr = np.zeros(4, dtype=np.float32)
        total = sum(w for _, w in pairs) or 1.0
        for k, (ji, wi) in enumerate(pairs[:4]):
            j_arr[k] = ji
            w_arr[k] = wi / total
        return j_arr, w_arr

    for r_idx in range(n_rings):
        cx, cy, cz = rings_centers[r_idx]
        ra, rb = rings_radii[r_idx]
        j_vec, w_vec = _encode_jw(rings_joints[r_idx])
        for s_idx in range(sides):
            if axis == "z":
                p = np.array([cx + ra * cos_a[s_idx], cy + rb * sin_a[s_idx], cz], dtype=np.float64)
            elif axis == "y":
                p = np.array([cx + ra * cos_a[s_idx], cy, cz + rb * sin_a[s_idx]], dtype=np.float64)
            else:
                p = np.array([cx, cy + ra * cos_a[s_idx], cz + rb * sin_a[s_idx]], dtype=np.float64)
            verts.append(p)
            jnts.append(j_vec)
            wgts.append(w_vec)

    tris: list[tuple[int, int, int]] = []
    for r_idx in range(n_rings - 1):
        row0 = r_idx * sides
        row1 = (r_idx + 1) * sides
        for s_idx in range(sides):
            sn = (s_idx + 1) % sides
            a = row0 + s_idx
            b = row0 + sn
            c = row1 + sn
            d = row1 + s_idx
            tris.append((a, b, c))
            tris.append((a, c, d))

    if cap_bottom:
        center_idx = len(verts)
        verts.append(np.array(rings_centers[0], dtype=np.float64))
        jb, wb = _encode_jw(rings_joints[0])
        jnts.append(jb)
        wgts.append(wb)
        for s_idx in range(sides):
            sn = (s_idx + 1) % sides
            tris.append((center_idx, sn, s_idx))

    if cap_top:
        center_idx = len(verts)
        verts.append(np.array(rings_centers[-1], dtype=np.float64))
        jt, wt = _encode_jw(rings_joints[-1])
        jnts.append(jt)
        wgts.append(wt)
        top_row = (n_rings - 1) * sides
        for s_idx in range(sides):
            sn = (s_idx + 1) % sides
            tris.append((center_idx, top_row + s_idx, top_row + sn))

    pos_arr = np.vstack(verts)
    tri_arr = np.asarray(tris, dtype=np.uint32)

    # Compute smooth outward vertex normals from face cross products
    normals = np.zeros_like(pos_arr)
    v0 = pos_arr[tri_arr[:, 0]]
    v1 = pos_arr[tri_arr[:, 1]]
    v2 = pos_arr[tri_arr[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    for k in range(3):
        np.add.at(normals, tri_arr[:, k], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normals = normals / norms

    return (
        pos_arr,
        normals,
        np.vstack(jnts),
        np.vstack(wgts),
        tri_arr.reshape(-1),
    )


def _uv_ellipsoid(
    center: Vec3,
    radii: Vec3,
    joint_weights: Sequence[tuple[int, float]],
    *,
    lat_steps: int = 12,
    lon_steps: int = 18,
    lat_min: float = -0.5 * math.pi,
    lat_max: float = 0.5 * math.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth UV ellipsoid (for head, hair cap, shoulders, hands)."""
    cx, cy, cz = center
    rx, ry, rz = radii
    lats = np.linspace(lat_min, lat_max, lat_steps)
    lons = np.linspace(0.0, 2.0 * math.pi, lon_steps, endpoint=False)

    j_vec = np.zeros(4, dtype=np.uint16)
    w_vec = np.zeros(4, dtype=np.float32)
    total = sum(w for _, w in joint_weights) or 1.0
    for k, (ji, wi) in enumerate(joint_weights[:4]):
        j_vec[k] = ji
        w_vec[k] = wi / total

    verts: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    for phi in lats:
        cp = math.cos(phi)
        sp = math.sin(phi)
        for theta in lons:
            ct = math.cos(theta)
            st = math.sin(theta)
            verts.append(np.array([cx + rx * cp * ct, cy + ry * cp * st, cz + rz * sp], dtype=np.float64))
            nx = (cp * ct) / max(rx, 1e-6)
            ny = (cp * st) / max(ry, 1e-6)
            nz = sp / max(rz, 1e-6)
            n = np.array([nx, ny, nz], dtype=np.float64)
            n_len = float(np.linalg.norm(n))
            norms.append(n / (n_len if n_len > 1e-12 else 1.0))

    tris: list[tuple[int, int, int]] = []
    for i in range(lat_steps - 1):
        r0 = i * lon_steps
        r1 = (i + 1) * lon_steps
        for j in range(lon_steps):
            jn = (j + 1) % lon_steps
            tris.append((r0 + j, r0 + jn, r1 + jn))
            tris.append((r0 + j, r1 + jn, r1 + j))

    n_v = len(verts)
    return (
        np.vstack(verts),
        np.vstack(norms),
        np.tile(j_vec, (n_v, 1)),
        np.tile(w_vec, (n_v, 1)),
        np.asarray(tris, dtype=np.uint32).reshape(-1),
    )


def _limb_segment(
    a: Vec3,
    b: Vec3,
    r_a: tuple[float, float],
    r_b: tuple[float, float],
    j_a: int,
    j_b: int,
    *,
    rings: int = 6,
    sides: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a smooth tapered limb segment from joint ``a`` to ``b`` with blended skin weights."""
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    axis_vec = pb - pa
    length = float(np.linalg.norm(axis_vec))
    u_ax = axis_vec / max(length, 1e-9)
    ref = np.array([0.0, 1.0, 0.0]) if abs(u_ax[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u_ax, ref)
    e1 /= float(np.linalg.norm(e1))
    e2 = np.cross(u_ax, e1)

    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    verts: list[np.ndarray] = []
    jnts: list[np.ndarray] = []
    wgts: list[np.ndarray] = []

    for i in range(rings):
        t = i / max(1, rings - 1)
        center = (1.0 - t) * pa + t * pb
        # Mild anatomical muscle fullness near 35% of segment
        bulge = 1.0 + 0.08 * math.sin(math.pi * t)
        rx = ((1.0 - t) * r_a[0] + t * r_b[0]) * bulge
        ry = ((1.0 - t) * r_a[1] + t * r_b[1]) * bulge
        j_vec = np.array([j_a, j_b, 0, 0], dtype=np.uint16)
        w_b = float(t)
        w_vec = np.array([1.0 - w_b, w_b, 0.0, 0.0], dtype=np.float32)
        for ang in angles:
            p = center + rx * math.cos(ang) * e1 + ry * math.sin(ang) * e2
            verts.append(p)
            jnts.append(j_vec)
            wgts.append(w_vec)

    tris: list[tuple[int, int, int]] = []
    for i in range(rings - 1):
        r0 = i * sides
        r1 = (i + 1) * sides
        for s in range(sides):
            sn = (s + 1) % sides
            tris.append((r0 + s, r0 + sn, r1 + sn))
            tris.append((r0 + s, r1 + sn, r1 + s))

    # End caps
    c0_idx = len(verts)
    verts.append(pa)
    jnts.append(np.array([j_a, 0, 0, 0], dtype=np.uint16))
    wgts.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    for s in range(sides):
        sn = (s + 1) % sides
        tris.append((c0_idx, sn, s))

    c1_idx = len(verts)
    verts.append(pb)
    jnts.append(np.array([j_b, 0, 0, 0], dtype=np.uint16))
    wgts.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    top_r = (rings - 1) * sides
    for s in range(sides):
        sn = (s + 1) % sides
        tris.append((c1_idx, top_r + s, top_r + sn))

    pos_arr = np.vstack(verts)
    tri_arr = np.asarray(tris, dtype=np.uint32)
    normals = np.zeros_like(pos_arr)
    fn = np.cross(pos_arr[tri_arr[:, 1]] - pos_arr[tri_arr[:, 0]], pos_arr[tri_arr[:, 2]] - pos_arr[tri_arr[:, 0]])
    for k in range(3):
        np.add.at(normals, tri_arr[:, k], fn)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(n_len < 1e-12, 1.0, n_len)

    return pos_arr, normals, np.vstack(jnts), np.vstack(wgts), tri_arr.reshape(-1)


def _assemble_human_submeshes(height_m: float = NOMINAL_HEIGHT_M) -> tuple[dict[str, Vec3], list[_SubMesh]]:
    """Construct complete clothed adult male human geometry across 5 PBR material groups."""
    joints = rest_joints_world(height_m=height_m, body_scale=1.0)
    j_idx = {name: i for i, name in enumerate(JOINT_NAMES)}

    # Material groups: 0=skin, 1=hair, 2=shirt, 3=trousers, 4=shoes
    skin = _SubMesh(0, [], [], [], [], [])
    hair = _SubMesh(1, [], [], [], [], [])
    shirt = _SubMesh(2, [], [], [], [], [])
    trousers = _SubMesh(3, [], [], [], [], [])
    shoes = _SubMesh(4, [], [], [], [], [])

    h = height_m

    # 1. Torso shirt (from waist/hips up to collar)
    torso_rings = [
        (0.0, 0.0, 0.500 * h),
        (0.0, 0.0, 0.560 * h),
        (0.0, 0.005 * h, 0.635 * h),
        (0.0, 0.010 * h, 0.715 * h),
        (0.0, 0.012 * h, 0.775 * h),
        (0.0, 0.004 * h, 0.822 * h),
        (0.0, 0.0, 0.840 * h),
    ]
    torso_radii = [
        (0.168, 0.112),
        (0.162, 0.108),
        (0.152, 0.105),
        (0.178, 0.118),
        (0.205, 0.122),
        (0.190, 0.098),
        (0.072, 0.068),
    ]
    torso_weights = [
        ((j_idx["pelvis"], 1.0),),
        ((j_idx["pelvis"], 0.7), (j_idx["spine"], 0.3)),
        ((j_idx["spine"], 1.0),),
        ((j_idx["spine"], 0.35), (j_idx["chest"], 0.65)),
        ((j_idx["chest"], 1.0),),
        ((j_idx["chest"], 0.75), (j_idx["neck"], 0.25)),
        ((j_idx["neck"], 1.0),),
    ]
    shirt.add(*_loft_tube(torso_rings, torso_radii, torso_weights, sides=20))

    # Pelvis / hips shorts-base in trouser material so hip joint never shows gaps
    pelvis_rings = [
        (0.0, 0.0, 0.455 * h),
        (0.0, 0.0, 0.495 * h),
        (0.0, 0.0, 0.530 * h),
    ]
    pelvis_radii = [
        (0.162, 0.105),
        (0.172, 0.115),
        (0.166, 0.110),
    ]
    pelvis_weights = [
        ((j_idx["pelvis"], 1.0),),
        ((j_idx["pelvis"], 1.0),),
        ((j_idx["pelvis"], 1.0),),
    ]
    trousers.add(*_loft_tube(pelvis_rings, pelvis_radii, pelvis_weights, sides=18))

    # 2. Neck, Head, Nose, Ears (skin) + Hair cap
    neck_rings = [
        (0.0, 0.0, 0.825 * h),
        (0.0, 0.002 * h, 0.855 * h),
        (0.0, 0.005 * h, 0.880 * h),
    ]
    neck_radii = [(0.060, 0.058), (0.056, 0.055), (0.058, 0.058)]
    neck_weights = [
        ((j_idx["neck"], 0.8), (j_idx["chest"], 0.2)),
        ((j_idx["neck"], 1.0),),
        ((j_idx["neck"], 0.4), (j_idx["head"], 0.6)),
    ]
    skin.add(*_loft_tube(neck_rings, neck_radii, neck_weights, sides=14))

    head_c = joints["head"]
    skin.add(*_uv_ellipsoid(head_c, (0.084, 0.102, 0.115), ((j_idx["head"], 1.0),), lat_steps=14, lon_steps=20))
    # Nose bridge facing +Y
    nose_c = (head_c[0], head_c[1] + 0.100, head_c[2] - 0.010)
    skin.add(*_uv_ellipsoid(nose_c, (0.014, 0.022, 0.024), ((j_idx["head"], 1.0),), lat_steps=6, lon_steps=10))
    # Ears
    for sign in (-1.0, 1.0):
        ear_c = (head_c[0] + sign * 0.084, head_c[1] - 0.006, head_c[2] - 0.005)
        skin.add(*_uv_ellipsoid(ear_c, (0.014, 0.020, 0.030), ((j_idx["head"], 1.0),), lat_steps=6, lon_steps=10))

    # Hair cap covering upper/back cranium
    hair_c = (head_c[0], head_c[1] - 0.008, head_c[2] + 0.018)
    hair.add(
        *_uv_ellipsoid(
            hair_c,
            (0.088, 0.106, 0.108),
            ((j_idx["head"], 1.0),),
            lat_steps=10,
            lon_steps=20,
            lat_min=-0.05 * math.pi,
            lat_max=0.5 * math.pi,
        )
    )

    # 3. Shoulders, Arms (shirt sleeves + forearms in skin), and Hands
    for side in ("left", "right"):
        sh = joints[f"{side}_shoulder"]
        el = joints[f"{side}_elbow"]
        wr = joints[f"{side}_wrist"]
        j_sh = j_idx[f"{side}_shoulder"]
        j_el = j_idx[f"{side}_elbow"]
        j_wr = j_idx[f"{side}_wrist"]

        # Rounded deltoid shoulder cap (shirt)
        shirt.add(*_uv_ellipsoid(sh, (0.068, 0.064, 0.062), ((j_sh, 0.75), (j_idx["chest"], 0.25)), lat_steps=8, lon_steps=12))

        # Upper arm split into sleeve (top 55%) and lower upper-arm skin (bottom 45%)
        mid_arm = tuple(0.45 * sh[k] + 0.55 * el[k] for k in range(3))
        shirt.add(*_limb_segment(sh, mid_arm, (0.058, 0.056), (0.052, 0.050), j_sh, j_el, rings=5, sides=12))
        skin.add(*_limb_segment(mid_arm, el, (0.048, 0.046), (0.043, 0.042), j_sh, j_el, rings=5, sides=12))
        # Elbow cap + Forearm
        skin.add(*_uv_ellipsoid(el, (0.044, 0.043, 0.044), ((j_el, 1.0),), lat_steps=6, lon_steps=10))
        skin.add(*_limb_segment(el, wr, (0.043, 0.042), (0.032, 0.030), j_el, j_wr, rings=6, sides=12))

        # Sculpted hand (palm + finger silhouette + thumb)
        hand_c = (wr[0], wr[1] + 0.008, wr[2] - 0.065)
        skin.add(*_uv_ellipsoid(hand_c, (0.022, 0.042, 0.068), ((j_wr, 1.0),), lat_steps=8, lon_steps=12))
        thumb_sign = 1.0 if side == "left" else -1.0
        thumb_c = (wr[0] + thumb_sign * 0.012, wr[1] + 0.042, wr[2] - 0.045)
        skin.add(*_uv_ellipsoid(thumb_c, (0.012, 0.016, 0.028), ((j_wr, 1.0),), lat_steps=6, lon_steps=8))

    # 4. Legs (trousers) & Sculpted Shoes/Feet
    for side in ("left", "right"):
        hp = joints[f"{side}_hip"]
        kn = joints[f"{side}_knee"]
        an = joints[f"{side}_ankle"]
        j_hp = j_idx[f"{side}_hip"]
        j_kn = j_idx[f"{side}_knee"]
        j_an = j_idx[f"{side}_ankle"]

        # Hip joint filler, thigh, knee cap, calf/shin
        trousers.add(*_uv_ellipsoid(hp, (0.086, 0.086, 0.075), ((j_hp, 0.6), (j_idx["pelvis"], 0.4)), lat_steps=8, lon_steps=12))
        trousers.add(*_limb_segment(hp, kn, (0.084, 0.086), (0.062, 0.064), j_hp, j_kn, rings=7, sides=14))
        trousers.add(*_uv_ellipsoid(kn, (0.063, 0.064, 0.062), ((j_kn, 1.0),), lat_steps=8, lon_steps=12))
        trousers.add(*_limb_segment(kn, an, (0.060, 0.062), (0.044, 0.046), j_kn, j_an, rings=7, sides=14))

        # Sculpted shoe/foot extending from heel (-Y) to toe (+Y) resting cleanly at z = 0
        shoe_rings = [
            (an[0], an[1] - 0.048, 0.036),
            (an[0], an[1] + 0.010, 0.040),
            (an[0], an[1] + 0.095, 0.034),
            (an[0], an[1] + 0.165, 0.024),
            (an[0], an[1] + 0.195, 0.016),
        ]
        shoe_radii = [
            (0.042, 0.036),
            (0.048, 0.040),
            (0.052, 0.034),
            (0.046, 0.024),
            (0.032, 0.016),
        ]
        shoe_weights = [((j_an, 1.0),) for _ in shoe_rings]
        shoes.add(*_loft_tube(shoe_rings, shoe_radii, shoe_weights, sides=14, axis="y"))

    return joints, [skin, hair, shirt, trousers, shoes]


def _world_array_to_gltf(arr: np.ndarray) -> np.ndarray:
    """Convert (N, 3) world Z-up coordinates/normals to glTF Y-up."""
    out = np.empty_like(arr, dtype=np.float32)
    out[:, 0] = arr[:, 0]
    out[:, 1] = arr[:, 2]
    out[:, 2] = -arr[:, 1]
    return out


def _pad4(buf: bytearray, pad_byte: bytes = b"\x00") -> None:
    rem = len(buf) % 4
    if rem:
        buf.extend(pad_byte * (4 - rem))


def build_procedural_human_glb(
    output_glb_path: str | Path,
    output_rig_json_path: str | Path | None = None,
    *,
    avatar_asset_id: str = AVATAR_ASSET_ID,
    height_m: float = NOMINAL_HEIGHT_M,
) -> dict[str, Any]:
    """Build and write a valid binary glTF 2.0 rigged human avatar and companion rig JSON."""
    joints_world, submeshes = _assemble_human_submeshes(height_m=height_m)
    joints_gltf = {name: world_to_gltf(pos) for name, pos in joints_world.items()}

    bin_buf = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def append_accessor(
        raw_bytes: bytes,
        *,
        component_type: int,
        count: int,
        acc_type: str,
        target: int | None = None,
        min_vals: list[float] | None = None,
        max_vals: list[float] | None = None,
    ) -> int:
        _pad4(bin_buf, b"\x00")
        byte_offset = len(bin_buf)
        bin_buf.extend(raw_bytes)
        bv: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(raw_bytes),
        }
        if target is not None:
            bv["target"] = target
        bv_idx = len(buffer_views)
        buffer_views.append(bv)

        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "componentType": component_type,
            "count": count,
            "type": acc_type,
        }
        if min_vals is not None:
            acc["min"] = min_vals
        if max_vals is not None:
            acc["max"] = max_vals
        acc_idx = len(accessors)
        accessors.append(acc)
        return acc_idx

    # Inverse bind matrices for the 17 joints (column-major 4x4 float32)
    ibm_list: list[np.ndarray] = []
    for name in JOINT_NAMES:
        jx, jy, jz = joints_gltf[name]
        # Column-major translation matrix T(-j)
        mat_col_major = np.array(
            [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                -jx, -jy, -jz, 1.0,
            ],
            dtype="<f4",
        )
        ibm_list.append(mat_col_major)
    ibm_bytes = np.concatenate(ibm_list).tobytes()
    ibm_accessor_idx = append_accessor(
        ibm_bytes,
        component_type=5126,  # FLOAT
        count=len(JOINT_NAMES),
        acc_type="MAT4",
    )

    # Mesh primitives (one per PBR material)
    primitives: list[dict[str, Any]] = []
    total_vertices = 0
    total_triangles = 0

    for sm in submeshes:
        if not sm.positions:
            continue
        pos_w = np.vstack(sm.positions)
        nrm_w = np.vstack(sm.normals)
        pos_g = _world_array_to_gltf(pos_w)
        nrm_g = _world_array_to_gltf(nrm_w)
        # Re-normalize normals cleanly
        n_len = np.linalg.norm(nrm_g, axis=1, keepdims=True)
        nrm_g = (nrm_g / np.where(n_len < 1e-12, 1.0, n_len)).astype("<f4")
        pos_g = pos_g.astype("<f4")

        jnt_arr = np.vstack(sm.joints).astype("<u2")
        wgt_arr = np.vstack(sm.weights).astype("<f4")
        # Guarantee strict normalization of skin weights to 1.0
        w_sum = wgt_arr.sum(axis=1, keepdims=True)
        wgt_arr = (wgt_arr / np.where(w_sum < 1e-12, 1.0, w_sum)).astype("<f4")

        idx_arr = np.concatenate(sm.indices)
        if pos_g.shape[0] < 65535:
            idx_typed = idx_arr.astype("<u2")
            idx_comp_type = 5123  # UNSIGNED_SHORT
        else:
            idx_typed = idx_arr.astype("<u4")
            idx_comp_type = 5125  # UNSIGNED_INT

        pos_min = [float(v) for v in pos_g.min(axis=0)]
        pos_max = [float(v) for v in pos_g.max(axis=0)]

        pos_acc = append_accessor(
            pos_g.tobytes(),
            component_type=5126,
            count=int(pos_g.shape[0]),
            acc_type="VEC3",
            target=34962,  # ARRAY_BUFFER
            min_vals=pos_min,
            max_vals=pos_max,
        )
        nrm_acc = append_accessor(
            nrm_g.tobytes(),
            component_type=5126,
            count=int(nrm_g.shape[0]),
            acc_type="VEC3",
            target=34962,
        )
        jnt_acc = append_accessor(
            jnt_arr.tobytes(),
            component_type=5123,  # UNSIGNED_SHORT
            count=int(jnt_arr.shape[0]),
            acc_type="VEC4",
            target=34962,
        )
        wgt_acc = append_accessor(
            wgt_arr.tobytes(),
            component_type=5126,
            count=int(wgt_arr.shape[0]),
            acc_type="VEC4",
            target=34962,
        )
        idx_acc = append_accessor(
            idx_typed.tobytes(),
            component_type=idx_comp_type,
            count=int(idx_typed.shape[0]),
            acc_type="SCALAR",
            target=34963,  # ELEMENT_ARRAY_BUFFER
        )

        primitives.append(
            {
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL": nrm_acc,
                    "JOINTS_0": jnt_acc,
                    "WEIGHTS_0": wgt_acc,
                },
                "indices": idx_acc,
                "material": sm.material_idx,
                "mode": 4,  # TRIANGLES
            }
        )
        total_vertices += int(pos_g.shape[0])
        total_triangles += int(idx_typed.shape[0] // 3)

    _pad4(bin_buf, b"\x00")

    # Joint nodes (0..16) with hierarchical local translations
    joint_idx_map = {name: i for i, name in enumerate(JOINT_NAMES)}
    children_map: dict[str, list[int]] = {name: [] for name in JOINT_NAMES}
    for name, parent in JOINT_PARENTS.items():
        if parent is not None:
            children_map[parent].append(joint_idx_map[name])

    nodes: list[dict[str, Any]] = []
    for name in JOINT_NAMES:
        parent = JOINT_PARENTS[name]
        pos = joints_gltf[name]
        if parent is None:
            rel = pos
        else:
            ppos = joints_gltf[parent]
            rel = (pos[0] - ppos[0], pos[1] - ppos[1], pos[2] - ppos[2])
        node: dict[str, Any] = {
            "name": name,
            "translation": [round(rel[0], 6), round(rel[1], 6), round(rel[2], 6)],
        }
        if children_map[name]:
            node["children"] = children_map[name]
        nodes.append(node)

    # Skinned mesh node (index 17)
    mesh_node_idx = len(nodes)
    nodes.append({"name": f"{avatar_asset_id}_mesh", "mesh": 0, "skin": 0})

    materials = [
        {
            "name": "human_skin",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.84, 0.64, 0.51, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.68,
            },
        },
        {
            "name": "human_hair",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.16, 0.12, 0.10, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.82,
            },
        },
        {
            "name": "casual_shirt",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.22, 0.43, 0.65, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.85,
            },
        },
        {
            "name": "trousers",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.18, 0.22, 0.29, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.86,
            },
        },
        {
            "name": "shoes",
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.13, 0.11, 0.10, 1.0],
                "metallicFactor": 0.05,
                "roughnessFactor": 0.55,
            },
        },
    ]

    gltf_doc: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "radarsim.motion.avatar (procedural skinned humanoid)",
            "copyright": "MIT / Project-generated procedural asset",
        },
        "scene": 0,
        "scenes": [{"name": avatar_asset_id, "nodes": [0, mesh_node_idx]}],
        "nodes": nodes,
        "skins": [
            {
                "name": f"{avatar_asset_id}_skin",
                "skeleton": 0,
                "joints": list(range(len(JOINT_NAMES))),
                "inverseBindMatrices": ibm_accessor_idx,
            }
        ],
        "meshes": [{"name": f"{avatar_asset_id}_human_body", "primitives": primitives}],
        "materials": materials,
        "buffers": [{"byteLength": len(bin_buf)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = bytearray(json.dumps(gltf_doc, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    _pad4(json_bytes, b" ")

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    glb = bytearray()
    # GLB header: magic ('glTF'), version (2), length
    glb.extend(struct.pack("<III", 0x46546C67, 2, total_length))
    # JSON chunk
    glb.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    glb.extend(json_bytes)
    # BIN chunk
    glb.extend(struct.pack("<II", len(bin_buf), 0x004E4942))
    glb.extend(bin_buf)

    glb_path = Path(output_glb_path)
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(bytes(glb))

    rig_spec: dict[str, Any] = {
        "schema_version": "1.0",
        "avatar_asset_id": avatar_asset_id,
        "glb_file": glb_path.name,
        "presentation": "adult_male_presenting_human_avatar",
        "nominal_height_m": height_m,
        "vertex_count": total_vertices,
        "triangle_count": total_triangles,
        "materials": [m["name"] for m in materials],
        "draw_skeleton": False,
        "hide_bones": True,
        "coordinate_convention": {
            "simulation_world": "right_handed_z_up_x_right_y_forward",
            "gltf_asset": "right_handed_y_up_x_right_minus_z_forward",
            "world_to_gltf_matrix": [list(row) for row in WORLD_TO_GLTF_MATRIX],
        },
        "joint_names": list(JOINT_NAMES),
        "joint_parents": dict(JOINT_PARENTS),
        "joint_node_indices": {name: i for i, name in enumerate(JOINT_NAMES)},
        "rest_joints_world_m": {k: [round(v[0], 6), round(v[1], 6), round(v[2], 6)] for k, v in joints_world.items()},
        "rest_joints_gltf_m": {k: [round(v[0], 6), round(v[1], 6), round(v[2], 6)] for k, v in joints_gltf.items()},
        "threejs_adapter_notes": (
            "Load GLB with Three.js GLTFLoader. Keep skeleton/bones invisible. "
            "Map each MotionFrame.joints_m[name] via world_to_gltf_matrix and update "
            "each bone's world position/orientation in hierarchical parent-to-child order."
        ),
        "license": "MIT (procedurally generated within repository; no third-party assets)",
    }

    if output_rig_json_path is not None:
        rig_path = Path(output_rig_json_path)
        rig_path.parent.mkdir(parents=True, exist_ok=True)
        rig_path.write_text(json.dumps(rig_spec, indent=2) + "\n", encoding="utf-8")

    return rig_spec


def validate_glb_asset(
    glb_path: str | Path,
    rig_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly parse and validate a binary glTF 2.0 rigged human avatar file."""
    path = Path(glb_path)
    if not path.exists():
        raise FileNotFoundError(f"GLB asset not found: {path}")
    data = path.read_bytes()
    if len(data) < 28:
        raise ValueError("GLB file is truncated")

    magic, version, declared_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Invalid GLB magic: 0x{magic:08x}")
    if version != 2:
        raise ValueError(f"Unsupported glTF version: {version}")
    if declared_length != len(data):
        raise ValueError(f"GLB declared length {declared_length} != file size {len(data)}")

    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("First GLB chunk must be JSON")
    json_start = 20
    json_end = json_start + json_len
    doc = json.loads(data[json_start:json_end].decode("utf-8"))

    bin_len, bin_type = struct.unpack_from("<II", data, json_end)
    if bin_type != 0x004E4942:
        raise ValueError("Second GLB chunk must be BIN")
    bin_data = data[json_end + 8 : json_end + 8 + bin_len]
    if len(bin_data) != bin_len:
        raise ValueError("GLB BIN chunk truncated")

    # Verify skin and all 17 INTERFACES.md joints exist
    if not doc.get("skins"):
        raise ValueError("GLB contains no skin rig")
    skin = doc["skins"][0]
    joint_indices = skin["joints"]
    node_names = [doc["nodes"][idx]["name"] for idx in joint_indices]
    if set(node_names) != set(JOINT_NAMES) or len(node_names) != 17:
        raise ValueError(f"GLB skin joints {node_names} do not match INTERFACES.md 17 joints")

    # Verify mesh primitives, vertex positions, normals, weights, and triangle indices
    primitives = doc["meshes"][0]["primitives"]
    if len(primitives) < 3:
        raise ValueError("Expected multi-material clothed human mesh primitives")

    total_verts = 0
    total_tris = 0
    bbox_min = np.array([math.inf, math.inf, math.inf])
    bbox_max = np.array([-math.inf, -math.inf, -math.inf])

    for prim in primitives:
        attrs = prim["attributes"]
        for req_attr in ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0"):
            if req_attr not in attrs:
                raise ValueError(f"Primitive missing attribute {req_attr}")

        pos_acc = doc["accessors"][attrs["POSITION"]]
        pos_bv = doc["bufferViews"][pos_acc["bufferView"]]
        pos = np.frombuffer(
            bin_data, dtype="<f4", count=pos_acc["count"] * 3, offset=pos_bv["byteOffset"]
        ).reshape(-1, 3)
        if not np.all(np.isfinite(pos)):
            raise ValueError("Non-finite vertex positions in GLB")
        bbox_min = np.minimum(bbox_min, pos.min(axis=0))
        bbox_max = np.maximum(bbox_max, pos.max(axis=0))

        wgt_acc = doc["accessors"][attrs["WEIGHTS_0"]]
        wgt_bv = doc["bufferViews"][wgt_acc["bufferView"]]
        wgt = np.frombuffer(
            bin_data, dtype="<f4", count=wgt_acc["count"] * 4, offset=wgt_bv["byteOffset"]
        ).reshape(-1, 4)
        if not np.allclose(wgt.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError("Skinning weights do not sum to 1.0")

        idx_acc = doc["accessors"][prim["indices"]]
        idx_bv = doc["bufferViews"][idx_acc["bufferView"]]
        idx_dtype = "<u2" if idx_acc["componentType"] == 5123 else "<u4"
        idx = np.frombuffer(
            bin_data, dtype=idx_dtype, count=idx_acc["count"], offset=idx_bv["byteOffset"]
        )
        if int(idx.max()) >= pos_acc["count"]:
            raise ValueError("Triangle index out of vertex bounds")

        total_verts += int(pos_acc["count"])
        total_tris += int(idx_acc["count"] // 3)

    if rig_json_path is not None:
        rig_data = json.loads(Path(rig_json_path).read_text(encoding="utf-8"))
        if rig_data.get("draw_skeleton") is not False:
            raise ValueError("Rig spec must enforce draw_skeleton=False")
        if set(rig_data.get("joint_names", [])) != set(JOINT_NAMES):
            raise ValueError("Rig JSON joint_names mismatch")

    return {
        "valid": True,
        "file_size_bytes": len(data),
        "joint_count": len(node_names),
        "material_count": len(doc.get("materials", [])),
        "primitive_count": len(primitives),
        "vertex_count": total_verts,
        "triangle_count": total_tris,
        "bbox_gltf_min_m": [float(v) for v in bbox_min],
        "bbox_gltf_max_m": [float(v) for v in bbox_max],
        "height_m": float(bbox_max[1] - bbox_min[1]),
    }
