"""Collision-aware 2D A* path planner consuming the normalized ``Scene`` from T1.

Never hardcodes room dimensions or furniture coordinates; builds its navigation
grid dynamically from ``Scene`` using full-body footprint clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np

from radarsim.scene import OccupancyGrid2D, Scene, SceneObject

XY = tuple[float, float]


@dataclass(frozen=True)
class FurnitureInteractionSpot:
    """Navigable standing/approach spot and target surface pose for a furniture interaction."""

    object_id: str
    object_kind: str
    approach_xy_m: XY
    approach_yaw_rad: float
    seat_or_lie_xy_m: XY
    seat_or_lie_yaw_rad: float
    surface_z_m: float


class AStarNavigator:
    """8-connected collision-aware A* path planner with corner-cutting prevention and smoothing."""

    def __init__(
        self,
        scene: Scene,
        *,
        resolution_m: float = 0.10,
        body_radius_m: float = 0.15,
        obstacle_clearance_m: float = 0.12,
    ) -> None:
        self.scene = scene
        self.resolution_m = float(resolution_m)
        self.body_radius_m = float(body_radius_m)
        self.obstacle_clearance_m = float(obstacle_clearance_m)
        self.total_clearance_m = self.body_radius_m + self.obstacle_clearance_m
        self.grid: OccupancyGrid2D = scene.build_occupancy_grid_2d(
            resolution_m=self.resolution_m,
            clearance_m=self.total_clearance_m,
        )

    def _xy_to_cell(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.argmin(np.abs(self.grid.x_centers_m - x)))
        iy = int(np.argmin(np.abs(self.grid.y_centers_m - y)))
        return ix, iy

    def _cell_to_xy(self, ix: int, iy: int) -> XY:
        return (float(self.grid.x_centers_m[ix]), float(self.grid.y_centers_m[iy]))

    def _nearest_free_cell(self, cell: tuple[int, int], max_radius_cells: int = 8) -> tuple[int, int] | None:
        nx, ny = self.grid.shape
        cx, cy = cell
        if not self.grid.blocked[cx, cy]:
            return cell
        best: tuple[float, int, int] | None = None
        for r in range(1, max_radius_cells + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    x, y = cx + dx, cy + dy
                    if 0 <= x < nx and 0 <= y < ny and not self.grid.blocked[x, y]:
                        d = math.hypot(dx, dy)
                        if best is None or d < best[0]:
                            best = (d, x, y)
            if best is not None:
                return (best[1], best[2])
        return None

    def is_segment_clear(
        self,
        a: XY,
        b: XY,
        *,
        clearance_m: float | None = None,
        allow_doorway_exterior: bool = False,
    ) -> bool:
        """Check whether the straight segment ``a -> b`` stays free with full-body clearance."""
        rad = self.total_clearance_m if clearance_m is None else float(clearance_m)
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(2, int(math.ceil(dist / max(0.04, 0.5 * self.resolution_m))))
        for k in range(steps + 1):
            u = k / steps
            px = (1.0 - u) * a[0] + u * b[0]
            py = (1.0 - u) * a[1] + u * b[1]
            if not self.scene.is_point_free_2d(
                px, py, radius_m=rad, allow_doorway_exterior=allow_doorway_exterior
            ):
                return False
        return True

    def plan_path(self, start_xy: XY, goal_xy: XY) -> list[XY] | None:
        """Compute a collision-free smoothed 2D waypoint path inside the room using A*."""
        if self.is_segment_clear(start_xy, goal_xy):
            return [start_xy, goal_xy]

        start_cell = self._nearest_free_cell(self._xy_to_cell(*start_xy))
        goal_cell = self._nearest_free_cell(self._xy_to_cell(*goal_xy))
        if start_cell is None or goal_cell is None:
            return None

        nx, ny = self.grid.shape
        blocked = self.grid.blocked

        open_heap: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start_cell[0], start_cell[1]))
        g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        closed: set[tuple[int, int]] = set()

        neighbors = (
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0)),
        )

        found = False
        while open_heap:
            _, cur_g, cx, cy = heapq.heappop(open_heap)
            cur = (cx, cy)
            if cur in closed:
                continue
            if cur == goal_cell:
                found = True
                break
            closed.add(cur)

            for dx, dy, step_cost in neighbors:
                nx_i, ny_i = cx + dx, cy + dy
                if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                    continue
                if blocked[nx_i, ny_i]:
                    continue
                # Prevent diagonal corner-cutting across blocked cells
                if dx != 0 and dy != 0:
                    if blocked[cx + dx, cy] or blocked[cx, cy + dy]:
                        continue
                nxt = (nx_i, ny_i)
                tentative_g = cur_g + step_cost
                if tentative_g < g_score.get(nxt, math.inf):
                    g_score[nxt] = tentative_g
                    came_from[nxt] = cur
                    h = math.hypot(goal_cell[0] - nx_i, goal_cell[1] - ny_i)
                    heapq.heappush(open_heap, (tentative_g + h, tentative_g, nx_i, ny_i))

        if not found:
            return None

        # Reconstruct grid cell path
        cells: list[tuple[int, int]] = [goal_cell]
        cur = goal_cell
        while cur in came_from:
            cur = came_from[cur]
            cells.append(cur)
        cells.reverse()

        raw_pts: list[XY] = [self._cell_to_xy(ix, iy) for ix, iy in cells]
        if self.is_segment_clear(start_xy, raw_pts[0]):
            raw_pts.insert(0, start_xy)
        if self.is_segment_clear(raw_pts[-1], goal_xy):
            raw_pts.append(goal_xy)

        return self._smooth_path(raw_pts)

    def _smooth_path(self, pts: Sequence[XY]) -> list[XY]:
        """Greedy line-of-sight string-pulling that preserves full-body collision clearance."""
        if len(pts) <= 2:
            return list(pts)
        smoothed: list[XY] = [pts[0]]
        idx = 0
        n = len(pts)
        while idx < n - 1:
            farthest = idx + 1
            for probe in range(n - 1, idx, -1):
                if self.is_segment_clear(pts[idx], pts[probe]):
                    farthest = probe
                    break
            smoothed.append(pts[farthest])
            idx = farthest
        return smoothed

    def free_cells_xy(self, *, min_wall_margin_m: float = 0.0) -> list[XY]:
        """List all navigable interior cell centres (optionally keeping extra wall margin)."""
        pts: list[XY] = []
        nx, ny = self.grid.shape
        for ix in range(nx):
            x = float(self.grid.x_centers_m[ix])
            if x < min_wall_margin_m or x > self.scene.width_x_m - min_wall_margin_m:
                continue
            for iy in range(ny):
                if self.grid.blocked[ix, iy]:
                    continue
                y = float(self.grid.y_centers_m[iy])
                if y < min_wall_margin_m or y > self.scene.depth_y_m - min_wall_margin_m:
                    continue
                pts.append((x, y))
        return pts

    def find_furniture_interaction(
        self,
        obj: SceneObject,
        *,
        mode: str = "sit",
    ) -> FurnitureInteractionSpot | None:
        """Find a collision-free approach pose alongside a chair, sofa, or bed."""
        cx, cy, _ = obj.box.center_m
        hx, hy, _ = obj.box.half_extents_m
        axes = obj.box.axes_2d()
        ux = axes[0]  # local +X
        uy = axes[1]  # local +Y
        standoff = self.total_clearance_m + 0.14

        # Candidate approach directions (outward from each of the 4 sides of the furniture)
        candidates: list[tuple[np.ndarray, float]] = [
            (-uy, hy + standoff),
            (+uy, hy + standoff),
            (+ux, hx + standoff),
            (-ux, hx + standoff),
        ]

        best: FurnitureInteractionSpot | None = None
        for outward, dist in candidates:
            ax = float(cx + outward[0] * dist)
            ay = float(cy + outward[1] * dist)
            if not self.scene.is_point_free_2d(ax, ay, radius_m=self.total_clearance_m):
                continue
            # Facing away from furniture center along outward normal (so sitting faces out)
            face_out_yaw = math.atan2(-float(outward[0]), float(outward[1]))
            if mode == "lie":
                # Lie centered on top of bed/sofa along its longer horizontal axis
                long_axis = ux if hx >= hy else uy
                lie_yaw = math.atan2(-float(long_axis[0]), float(long_axis[1]))
                seat_xy = (float(cx), float(cy))
                seat_yaw = lie_yaw
            else:
                # Sit near edge of chair/sofa/bed facing outward
                seat_offset = max(0.0, (hy if abs(float(np.dot(outward, uy))) > 0.5 else hx) - 0.22)
                seat_xy = (
                    float(cx + outward[0] * seat_offset),
                    float(cy + outward[1] * seat_offset),
                )
                seat_yaw = face_out_yaw

            best = FurnitureInteractionSpot(
                object_id=obj.object_id,
                object_kind=obj.kind,
                approach_xy_m=(ax, ay),
                approach_yaw_rad=face_out_yaw,
                seat_or_lie_xy_m=seat_xy,
                seat_or_lie_yaw_rad=seat_yaw,
                surface_z_m=obj.box.z_max_m,
            )
            break
        return best
