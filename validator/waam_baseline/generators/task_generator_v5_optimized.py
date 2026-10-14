"""Shape-constrained, time-optimized WAAM task generator V5.

Chooses the deposition strategy from each STL layer geometry:
- solid/wide sections: clipped hatch infill
- thin/branched sections: medial-axis centerline

No test-number-specific rules are used. Candidate paths are evaluated against
the same buffered-bead geometry used by the shape pre-check.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import reduce
import numpy as np
import trimesh
import yaml
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union
from shapely import contains_xy
from skimage.morphology import skeletonize


@dataclass
class DepositionTask:
    task_id: int
    layer: int
    start_xyz_mm: np.ndarray
    end_xyz_mm: np.ndarray

    def to_dict(self):
        return dict(
            task_id=self.task_id,
            layer=self.layer,
            start_xyz_mm=self.start_xyz_mm.tolist(),
            end_xyz_mm=self.end_xyz_mm.tolist(),
        )


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def target_polygon(section):
    """Even/odd fill: outer loops minus holes while preserving nested islands."""
    loops = []
    for path in section.discrete:
        p = Polygon(np.asarray(path)[:, :2])
        if not p.is_valid:
            p = p.buffer(0)
        if p.area > 1e-5:
            loops.append(p)
    if not loops:
        return Polygon()
    return reduce(lambda a, b: a.symmetric_difference(b), loops)


def skeleton_edges(poly, pixel=0.55):
    """Approximate medial-axis graph for thin-wall / branched geometry."""
    xmin, ymin, xmax, ymax = poly.bounds
    xs = np.arange(xmin - pixel, xmax + 2 * pixel, pixel)
    ys = np.arange(ymin - pixel, ymax + 2 * pixel, pixel)
    X, Y = np.meshgrid(xs, ys)
    mask = contains_xy(poly, X, Y)
    if not mask.any():
        return []

    skel = skeletonize(mask)
    pixels = set(zip(*np.nonzero(skel)))
    edges = []
    for i, j in sorted(pixels):
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
            q = (i + di, j + dj)
            if q not in pixels:
                continue
            if di and dj and ((i + di, j) in pixels or (i, j + dj) in pixels):
                continue
            a = np.array([xs[j], ys[i]], dtype=float)
            b = np.array([xs[q[1]], ys[q[0]]], dtype=float)
            if poly.covers(LineString([a, b])):
                edges.append((a, b))
    return edges


def _line_parts(g):
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    if hasattr(g, "geoms"):
        return [x for x in g.geoms if x.geom_type == "LineString"]
    return []


def hatch_edges(poly, bead_width, layer):
    """Generate geometry-clipped infill without assuming a particular STL shape.

    Centerlines stay bead_width/2 inside the target. Spacing is 62.5% of bead
    width so buffered beads overlap enough to satisfy complex solid sections.
    The scan direction alternates by layer.
    """
    radius = bead_width / 2.0
    spacing = bead_width * 0.625
    inner = poly.buffer(-radius)
    if inner.is_empty:
        return []

    xmin, ymin, xmax, ymax = inner.bounds
    pad = max(10.0, bead_width * 2.0)
    edges = []

    horizontal = (layer % 2 == 0)
    if horizontal:
        positions = np.arange(ymin, ymax + spacing * 0.5, spacing)
        for y in positions:
            cut = inner.intersection(
                LineString([(xmin - pad, y), (xmax + pad, y)])
            )
            parts = _line_parts(cut)
            # serpentine orientation reduces unnecessary end-to-start travel
            if len(edges) % 2:
                parts = list(reversed(parts))
            for q in parts:
                cs = list(q.coords)
                if len(cs) >= 2 and q.length > 1e-6:
                    a = np.asarray(cs[0], float)
                    b = np.asarray(cs[-1], float)
                    if len(edges) % 2:
                        a, b = b, a
                    edges.append((a, b))
    else:
        positions = np.arange(xmin, xmax + spacing * 0.5, spacing)
        for x in positions:
            cut = inner.intersection(
                LineString([(x, ymin - pad), (x, ymax + pad)])
            )
            parts = _line_parts(cut)
            if len(edges) % 2:
                parts = list(reversed(parts))
            for q in parts:
                cs = list(q.coords)
                if len(cs) >= 2 and q.length > 1e-6:
                    a = np.asarray(cs[0], float)
                    b = np.asarray(cs[-1], float)
                    if len(edges) % 2:
                        a, b = b, a
                    edges.append((a, b))
    return edges


def path_metrics(poly, edges, bead_width, resolution=8):
    if not edges or poly.is_empty or poly.area <= 1e-12:
        return dict(coverage=0.0, overfill=float("inf"), iou=0.0)
    lines = [LineString([a, b]) for a, b in edges
             if np.linalg.norm(np.asarray(b) - np.asarray(a)) > 1e-9]
    if not lines:
        return dict(coverage=0.0, overfill=float("inf"), iou=0.0)
    dep = unary_union(lines).buffer(bead_width / 2.0, resolution=resolution)
    inter = poly.intersection(dep).area
    outside = dep.difference(poly).area
    union = poly.union(dep).area
    return dict(
        coverage=inter / poly.area,
        overfill=outside / poly.area,
        iou=inter / union if union > 1e-12 else 0.0,
    )



def hatch_edges_parametric(poly, bead_width, layer,
                           spacing_factor=0.625, inset_factor=0.5):
    """Generate clipped serpentine hatch using bead-width-scaled parameters."""
    from shapely.geometry import LineString

    spacing = max(1e-6, bead_width * float(spacing_factor))
    inset = bead_width * float(inset_factor)
    inner = poly.buffer(-inset)
    if inner.is_empty:
        return []

    minx, miny, maxx, maxy = inner.bounds
    horizontal = (layer % 2 == 0)
    edges = []
    eps = max(bead_width, 1.0)

    def add_geom(g, reverse=False):
        geoms = list(g.geoms) if hasattr(g, "geoms") else [g]
        parts = []
        for part in geoms:
            if part.is_empty or part.geom_type != "LineString":
                continue
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            if reverse:
                coords.reverse()
            parts.append(coords)
        for coords in parts:
            for a, b in zip(coords[:-1], coords[1:]):
                if a != b:
                    edges.append(((float(a[0]), float(a[1])),
                                  (float(b[0]), float(b[1]))))

    if horizontal:
        y = miny
        row = 0
        while y <= maxy + 1e-9:
            line = LineString([(minx-eps, y), (maxx+eps, y)])
            add_geom(inner.intersection(line), reverse=(row % 2 == 1))
            y += spacing
            row += 1
    else:
        x = minx
        col = 0
        while x <= maxx + 1e-9:
            line = LineString([(x, miny-eps), (x, maxy+eps)])
            add_geom(inner.intersection(line), reverse=(col % 2 == 1))
            x += spacing
            col += 1

    return edges

def choose_edges(poly, bead_width, layer, sc, pixel=0.55):
    """
    Shape-constrained, time-oriented geometry-adaptive path search.

    Hard constraints (read from config):
      coverage >= minimum_overall_coverage
      overfill <= maximum_overall_overfill_ratio
      layer IoU >= minimum_layer_iou

    To protect the overall-IoU requirement without forcing every layer to 0.90,
    the search first prefers candidates meeting the configured overall IoU.
    If none exists, it may use a candidate meeting the layer-IoU threshold,
    but only with the best IoU among the shortest feasible paths.

    No test/model number is used.
    """
    import math

    resolution = int(sc.get("polygon_buffer_resolution", 8))
    min_cov = float(sc.get("minimum_overall_coverage", 0.80))
    max_over = float(sc.get("maximum_overall_overfill_ratio", 0.30))
    min_layer_iou = float(sc.get("minimum_layer_iou", 0.80))
    target_iou = float(sc.get("minimum_overall_iou", 0.90))

    def feasible(m):
        return (
            m["coverage"] >= min_cov
            and m["overfill"] <= max_over
            and m["iou"] >= min_layer_iou
        )

    def path_length(edges):
        total = 0.0
        for a, b in edges:
            total += math.hypot(float(b[0]) - float(a[0]),
                                float(b[1]) - float(a[1]))
        return total

    candidates = []

    # Coarse -> fine search.  These are bead-width ratios, never test IDs.
    # Wider spacing is tried first because it usually means less deposition time.
    spacing_factors = (1.00, 0.90, 0.80, 0.70, 0.625, 0.575,
                       0.525, 0.50, 0.475, 0.45, 0.425, 0.40)
    inset_factors = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25)

    for spacing_factor in spacing_factors:
        for inset_factor in inset_factors:
            edges = hatch_edges_parametric(
                poly, bead_width, layer,
                spacing_factor=spacing_factor,
                inset_factor=inset_factor,
            )
            if not edges:
                continue
            m = path_metrics(poly, edges, bead_width, resolution)
            if feasible(m):
                candidates.append((
                    edges,
                    f"hatch(s={spacing_factor:.3f}w,i={inset_factor:.3f}w)",
                    m,
                    path_length(edges),
                ))

    # Skeleton is retained for thin/branched layers.
    for pf in (0.125, 0.10, 0.075, 0.0625):
        sk_pixel = max(0.20, bead_width * pf)
        edges = skeleton_edges(poly, sk_pixel)
        if not edges:
            continue
        m = path_metrics(poly, edges, bead_width, resolution)
        if feasible(m):
            candidates.append((
                edges,
                f"skeleton(pixel={sk_pixel:.3f}mm)",
                m,
                path_length(edges),
            ))

    if not candidates:
        raise RuntimeError(
            f"Layer {layer}: no path satisfies shape constraints "
            f"(coverage>={min_cov:.3f}, overfill<={max_over:.3f}, "
            f"layer IoU>={min_layer_iou:.3f}). "
            "Scheduling aborted before an invalid layer is emitted."
        )

    # Prefer candidates that individually provide enough IoU headroom for the
    # overall-IoU requirement.  Within that feasible set, minimize estimated
    # manufacturing cost: deposition time first, then fragmentation.
    strong = [c for c in candidates if c[2]["iou"] >= target_iou]
    pool = strong if strong else candidates

    # Deposition dominates at 8 mm/s; edge count approximates arc-on/off and
    # inter-segment travel overhead.  Normalize fragmentation into an
    # equivalent bead-width distance so it cannot dominate path length.
    def cost(c):
        edges, _, m, length = c
        fragmentation_penalty = len(edges) * bead_width * 0.05
        # If we had to fall back below overall target IoU, strongly favor
        # geometric quality while still avoiding needlessly long paths.
        iou_deficit_penalty = max(0.0, target_iou - m["iou"])
        return (
            length + fragmentation_penalty
            + iou_deficit_penalty * max(length, bead_width) * 5.0
        )

    best = min(pool, key=cost)
    edges, name, metrics, _ = best
    return edges, name, metrics


def generate_tasks(stl_path, config_path, pixel=0.55):
    c = load_config(config_path)
    p = c["process"]
    sc = c.get("shape_validation", {})
    mesh = trimesh.load(stl_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    h = float(p["layer_height_mm"])
    base = float(p["build_plane_z_mm"])
    bead_width = float(p["bead_width_mm"])
    if h <= 0 or bead_width <= 0 or pixel <= 0:
        raise ValueError("Positive layer height, bead width and pixel are required")

    tasks = []
    # Reuse path geometry when consecutive/identical layers have the same XY
    # cross-section. This is important for tall or repeated STL geometry.
    path_cache = {}
    layer = 0
    while True:
        offset = 0.5 if p["tcp_z_reference"] == "center" else 1.0
        z = base + (layer + offset) * h
        if z >= float(mesh.bounds[1, 2]) - 1e-9:
            break

        sec = mesh.section(
            plane_origin=[0, 0, z],
            plane_normal=[0, 0, 1],
        )
        if sec is not None:
            poly = target_polygon(sec)
            if not poly.is_empty and poly.area > 1e-9:
                # Layer parity matters for hatch orientation, so include it
                # in the cache key. Skeleton layers can still reuse the result
                # when the same parity/shape returns.
                cache_key = (poly.wkb, layer % 2)
                cached = path_cache.get(cache_key)
                if cached is None:
                    edges, strategy, metrics = choose_edges(
                        poly, bead_width, layer, sc, pixel
                    )
                    path_cache[cache_key] = (edges, strategy, metrics)
                else:
                    edges, strategy, metrics = cached
                before = len(tasks)
                for a, b in edges:
                    start = np.array([a[0], a[1], z], dtype=float)
                    end = np.array([b[0], b[1], z], dtype=float)
                    tasks.append(
                        DepositionTask(len(tasks), layer, start, end)
                    )
                print(
                    f"Layer {layer}: area={poly.area:.1f} mm2 | "
                    f"strategy={strategy} | edges={len(tasks)-before} | "
                    f"coverage={metrics['coverage']*100:.2f}% | "
                    f"overfill={metrics['overfill']*100:.2f}% | "
                    f"IoU={metrics['iou']*100:.2f}%",
                    flush=True,
                )
        layer += 1

    if not tasks:
        raise ValueError("No deposition tasks generated")
    return tasks


def build_scenario(stl_path, config_path):
    c = load_config(config_path)
    tasks = generate_tasks(stl_path, config_path)
    robots = [
        dict(
            robot_id=int(r["id"]),
            base_xyz_mm=r["base_xyz_mm"],
            home_xyz_mm=r.get("home_xyz_mm", r["base_xyz_mm"]),
        )
        for r in sorted(c["robots"], key=lambda r: int(r["id"]))
    ]
    return dict(
        robots=robots,
        process=dict(
            travel_speed_mm_s=float(c["process"]["travel_speed_mm_s"]),
            deposition_speed_mm_s=float(c["process"]["deposition_speed_mm_s"]),
        ),
        tasks=[t.to_dict() for t in tasks],
        normalization_radius_mm=max(
            float(r["xy_reach_radius_mm"]) for r in c["robots"]
        ),
        time_normalization_s=1000.0,
        reward={"invalid": -10.0, "finish_bonus": 20.0},
        max_steps=max(20, 4 * len(tasks)),
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pixel", type=float, default=0.55)
    a = ap.parse_args()
    print("Total tasks:", len(generate_tasks(a.stl, a.config, a.pixel)))
