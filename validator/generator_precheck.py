#!/usr/bin/env python
"""Fast generator-only shape precheck.

Uses the same task_generator_v2.py that safe_scheduler_v12.py imports.
It does NOT run robot scheduling/collision simulation.

Example:
  python generator_precheck.py --stl "../../test_data/03/target.stl" --config "../../test_data/03/config.yaml"
"""

import argparse
import math
import yaml
from shapely.geometry import LineString
from shapely.ops import unary_union

from task_generator_v2 import build_scenario


def xyz(p):
    if hasattr(p, "x"):
        return float(p.x), float(p.y), float(p.z)
    return float(p[0]), float(p[1]), float(p[2])


def get_task_points(task):
    # Support both dict tasks and object/dataclass tasks.
    pairs = (
        ("start", "end"),
        ("start_xyz", "end_xyz"),
        ("start_xyz_mm", "end_xyz_mm"),
        ("p0", "p1"),
        ("start_point", "end_point"),
    )

    if isinstance(task, dict):
        for a, b in pairs:
            if a in task and b in task:
                return xyz(task[a]), xyz(task[b])

        # Some generators store one segment under a nested geometry/segment dict.
        for container_key in ("segment", "geometry", "path"):
            nested = task.get(container_key)
            if isinstance(nested, dict):
                for a, b in pairs:
                    if a in nested and b in nested:
                        return xyz(nested[a]), xyz(nested[b])

        raise AttributeError(
            "Cannot find task endpoints on dict. "
            f"Available keys: {list(task.keys())}"
        )

    for a, b in pairs:
        if hasattr(task, a) and hasattr(task, b):
            return xyz(getattr(task, a)), xyz(getattr(task, b))

    raise AttributeError(
        f"Cannot find task endpoints on {type(task).__name__}. "
        f"Available fields: {list(vars(task).keys()) if hasattr(task, '__dict__') else 'unknown'}"
    )


def scenario_tasks(scenario):
    if hasattr(scenario, "tasks"):
        return list(scenario.tasks)
    if isinstance(scenario, dict) and "tasks" in scenario:
        return list(scenario["tasks"])
    raise AttributeError("build_scenario() result has no 'tasks' collection.")


def load_target_helpers():
    # Reuse the exact polygon extraction from the active generator, so the
    # precheck and scheduler generator cannot disagree about layer geometry.
    import task_generator_v2 as tg
    if not hasattr(tg, "target_polygon"):
        raise AttributeError("task_generator_v2.py has no target_polygon()")
    return tg.target_polygon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    process = cfg["process"]
    sc = cfg["shape_validation"]
    layer_h = float(process["layer_height_mm"])
    bead_w = float(process["bead_width_mm"])
    dep_speed = float(process["deposition_speed_mm_s"])
    resolution = int(sc.get("polygon_buffer_resolution", 8))

    min_cov = float(sc["minimum_overall_coverage"])
    max_over = float(sc["maximum_overall_overfill_ratio"])
    min_overall_iou = float(sc["minimum_overall_iou"])
    min_layer_iou = float(sc["minimum_layer_iou"])
    max_failed_ratio = float(sc["maximum_failed_layer_ratio"])

    # Build tasks using exactly the active generator.
    scenario = build_scenario(args.stl, args.config)
    tasks = scenario_tasks(scenario)

    # Load mesh only for layer target polygons.
    import trimesh
    mesh = trimesh.load(args.stl, force="mesh")
    target_polygon = load_target_helpers()

    # Group generated deposition segments by layer center Z.
    by_layer = {}
    total_path = 0.0
    for t in tasks:
        a, b = get_task_points(t)
        z = 0.5 * (a[2] + b[2])
        li = int(round((z - layer_h / 2.0) / layer_h))
        by_layer.setdefault(li, []).append((a, b))
        total_path += math.dist(a, b)

    zmin = float(mesh.bounds[0][2])
    zmax = float(mesh.bounds[1][2])
    n_layers = int(math.ceil((zmax - zmin) / layer_h - 1e-12))

    failed = 0
    sum_target = sum_inter = sum_dep = 0.0
    min_layer_cov = 1.0
    max_layer_over = 0.0
    min_liou = 1.0

    print("\n=== GENERATOR PRECHECK ===")
    print(f"STL: {args.stl}")
    print(f"Tasks: {len(tasks)}")

    for li in range(n_layers):
        z = zmin + (li + 0.5) * layer_h
        section = mesh.section(
            plane_origin=[0.0, 0.0, z],
            plane_normal=[0.0, 0.0, 1.0],
        )
        if section is None:
            continue

        poly = target_polygon(section)
        if poly is None or poly.is_empty:
            continue

        segs = by_layer.get(li, [])
        lines = [
            LineString([(a[0], a[1]), (b[0], b[1])])
            for a, b in segs
            if math.hypot(b[0]-a[0], b[1]-a[1]) > 1e-9
        ]
        dep = (
            unary_union(lines).buffer(bead_w / 2.0, resolution=resolution)
            if lines else poly.buffer(0).difference(poly.buffer(0))
        )

        ta = float(poly.area)
        inter = float(poly.intersection(dep).area)
        da = float(dep.area)
        union = ta + da - inter

        cov = inter / ta if ta > 0 else 1.0
        over = max(0.0, da - inter) / ta if ta > 0 else 0.0
        iou = inter / union if union > 0 else 1.0

        ok = cov >= min_cov and over <= max_over and iou >= min_layer_iou
        failed += 0 if ok else 1

        min_layer_cov = min(min_layer_cov, cov)
        max_layer_over = max(max_layer_over, over)
        min_liou = min(min_liou, iou)
        sum_target += ta
        sum_inter += inter
        sum_dep += da

        print(
            f"Layer {li:03d}: "
            f"coverage={100*cov:6.2f}% | "
            f"overfill={100*over:6.2f}% | "
            f"IoU={100*iou:6.2f}% | "
            f"{'PASS' if ok else 'FAIL'}"
        )

    overall_cov = sum_inter / sum_target if sum_target else 1.0
    overall_over = max(0.0, sum_dep - sum_inter) / sum_target if sum_target else 0.0
    overall_union = sum_target + sum_dep - sum_inter
    overall_iou = sum_inter / overall_union if overall_union else 1.0
    checked_layers = max(1, n_layers)
    failed_ratio = failed / checked_layers

    overall_ok = (
        overall_cov >= min_cov
        and overall_over <= max_over
        and overall_iou >= min_overall_iou
        and failed_ratio <= max_failed_ratio
    )

    print("\n=== SUMMARY ===")
    print(f"Layers           : {n_layers}")
    print(f"Failed layers    : {failed}/{n_layers} ({100*failed_ratio:.2f}%)")
    print(f"Min layer cov.   : {100*min_layer_cov:.2f}%")
    print(f"Max layer over.  : {100*max_layer_over:.2f}%")
    print(f"Min layer IoU    : {100*min_liou:.2f}%")
    print(f"Overall coverage : {100*overall_cov:.2f}%")
    print(f"Overall overfill : {100*overall_over:.2f}%")
    print(f"Overall IoU      : {100*overall_iou:.2f}%")
    print(f"Path length      : {total_path:.2f} mm")
    print(f"Estimated D time : {total_path/dep_speed:.2f} s")
    print(f"RESULT            : {'PASS' if overall_ok else 'FAIL'}")

    if not overall_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
