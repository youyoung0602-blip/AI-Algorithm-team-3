"""V15: adaptive geometry-derived bundle scheduler for fragmented WAAM paths.

Goals
-----
* Keep every deposition centerline exactly unchanged.
* Reduce repeated HOME -> tiny chain -> HOME travel by joining nearby chains into
  same-layer bundles with direct T connectors.
* Keep V12 collision checking, refined earliest-safe-start, 0.05 s boundary
  buffer, reach checks, PPO robot preference, recursive fallback, and final audit.
* No test-number-specific behavior.
* First try the proven V14 bundling. If that schedule under-utilizes robots that
  can geometrically contribute, rebuild bundles along robot-reachability regions
  and reschedule from scratch. This preserves V14 behavior when it already
  parallelizes well (e.g. Test03) while fixing single-robot monopolies.

Requires beside this file:
    task_generator_v2.py
    safe_scheduler_v12.py
    safe_scheduler_v2.py
    export_and_check_v2.py
    algorithm/environment/waam_env.py
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import csv
import math

import numpy as np
import yaml
from stable_baselines3 import PPO

from algorithm.environment.waam_env import WAAMBaselineEnv
from task_generator_v2 import build_scenario
from safe_scheduler_v2 import xyz, shifted, candidate_conflict
from export_and_check_v2 import audit
from safe_scheduler_v12 import (
    chain_tasks,
    chain_points,
    preferred_robot_for_chain,
    refine_earliest_collision_free_start,
)


def write_validator_csv(tracks, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["robot_id", "time_s", "x_mm", "y_mm", "z_mm", "mode"])
        for rid, segments in tracks.items():
            if not segments:
                raise ValueError(f"Robot {rid} has no segments")
            for t0, t1, a, b, mode in segments:
                w.writerow([
                    rid, f"{t0:.8f}",
                    f"{a[0]:.8f}", f"{a[1]:.8f}", f"{a[2]:.8f}", mode,
                ])
            last = segments[-1]
            b = last[3]
            w.writerow([
                rid, f"{last[1]:.8f}",
                f"{b[0]:.8f}", f"{b[1]:.8f}", f"{b[2]:.8f}", "W",
            ])
    return out


def chain_layer(chain, tasks):
    layers = {int(tasks[idx].get("layer", tasks[idx].get("layer_index", -1))) for idx, _ in chain}
    if len(layers) != 1:
        raise ValueError("A continuous chain crosses layers; refusing to bundle it")
    return next(iter(layers))


def chain_start_end(chain, tasks):
    pts, _ = chain_points(chain, tasks)
    return np.asarray(pts[0], float), np.asarray(pts[-1], float)


def robot_can_reach_points(robot, points):
    base = xyz(robot, "base_xyz_mm")
    reach = float(robot["xy_reach_radius_mm"])
    return all(np.linalg.norm(np.asarray(p)[:2] - base[:2]) <= reach + 1e-7 for p in points)


def direct_connector_is_useful(chain_a, chain_b, tasks, robots):
    """Geometry/config-derived merge rule, with no test-specific distance cutoff.

    Merge is allowed when at least one robot can reach both chains and the direct
    connector is shorter than that robot's HOME detour from A.end to B.start.
    """
    _sa, ea = chain_start_end(chain_a, tasks)
    sb, _eb = chain_start_end(chain_b, tasks)
    direct = float(np.linalg.norm(sb - ea))
    for r in robots:
        pts_a, _ = chain_points(chain_a, tasks)
        pts_b, _ = chain_points(chain_b, tasks)
        if not robot_can_reach_points(r, pts_a + pts_b):
            continue
        home = xyz(r, "home_xyz_mm")
        via_home = float(np.linalg.norm(ea - home) + np.linalg.norm(sb - home))
        if direct + 1e-9 < via_home:
            return True
    return False


def build_same_layer_bundles(chains, tasks, robots):
    """Greedy nearest-neighbour bundles, independently inside each layer.

    Complexity is sum(layer_chain_count^2), not all-chains^2.  This is suitable
    for tall models where each layer has hundreds of fragmented edges.
    """
    by_layer = defaultdict(list)
    for c in chains:
        by_layer[chain_layer(c, tasks)].append(c)

    bundles = []
    for layer in sorted(by_layer):
        layer_chains = by_layer[layer]
        remaining = set(range(len(layer_chains)))
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            bundle = [layer_chains[seed]]
            current = layer_chains[seed]

            while remaining:
                _s, end = chain_start_end(current, tasks)
                # Nearest next chain start.  Per-layer search avoids the O(N^2)
                # blow-up over all 64k+ chains.
                nearest = min(
                    remaining,
                    key=lambda j: float(np.linalg.norm(chain_start_end(layer_chains[j], tasks)[0] - end)),
                )
                nxt = layer_chains[nearest]
                if not direct_connector_is_useful(current, nxt, tasks, robots):
                    break
                bundle.append(nxt)
                remaining.remove(nearest)
                current = nxt

            bundles.append(bundle)
    return bundles



def chain_reachable_robots(chain, tasks, robots):
    pts, _ = chain_points(chain, tasks)
    return tuple(
        int(r["id"]) for r in robots
        if robot_can_reach_points(r, pts)
    )


def adaptive_reach_bundles(base_bundles, tasks, robots):
    """Split V14 bundles at geometry-derived reachability-region boundaries.

    A bundle must still stay on one layer.  No test IDs, fixed chain counts, or
    arbitrary spatial radii are used: boundaries come only from which configured
    robots can reach each unchanged deposition chain.
    """
    out = []
    for bundle in base_bundles:
        if not bundle:
            continue
        current = [bundle[0]]
        current_sig = chain_reachable_robots(bundle[0], tasks, robots)
        if not current_sig:
            raise RuntimeError("A deposition chain is unreachable by every robot")
        for chain in bundle[1:]:
            sig = chain_reachable_robots(chain, tasks, robots)
            if not sig:
                raise RuntimeError("A deposition chain is unreachable by every robot")
            # A change in feasible robot set is a natural spatial/reach boundary.
            # Splitting here lets different robots own different local regions.
            if sig != current_sig:
                out.append(current)
                current = [chain]
                current_sig = sig
            else:
                current.append(chain)
        out.append(current)
    return out


def geometrically_capable_robots(chains, tasks, robots):
    capable = set()
    for chain in chains:
        capable.update(chain_reachable_robots(chain, tasks, robots))
    return capable


def bundle_edge_ids(bundle, tasks):
    ids = []
    for chain in bundle:
        _pts, edge_ids = chain_points(chain, tasks)
        ids.extend(edge_ids)
    return ids


def bundle_segments(robot, bundle, tasks, config):
    """HOME -> D-chain -> direct T -> D-chain ... -> HOME."""
    home = xyz(robot, "home_xyz_mm")
    base = xyz(robot, "base_xyz_mm")
    reach = float(robot["xy_reach_radius_mm"])
    speed_t = float(config["process"]["travel_speed_mm_s"])
    speed_d = float(config["process"]["deposition_speed_mm_s"])
    if min(speed_t, speed_d) <= 0:
        raise ValueError("Speeds must be positive")

    chain_pts = [chain_points(c, tasks)[0] for c in bundle]
    all_pts = [home] + [p for pts in chain_pts for p in pts]
    for p in all_pts:
        if np.linalg.norm(np.asarray(p)[:2] - base[:2]) > reach + 1e-7:
            raise ValueError("XY reach exceeded")

    result = []
    t = 0.0

    def add(a, b, mode, speed):
        nonlocal t
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        dt = float(np.linalg.norm(b - a)) / speed
        if dt > 1e-10:
            result.append((t, t + dt, a.copy(), b.copy(), mode))
            t += dt

    add(home, chain_pts[0][0], "T", speed_t)
    for ci, pts in enumerate(chain_pts):
        for a, b in zip(pts[:-1], pts[1:]):
            add(a, b, "D", speed_d)
        if ci + 1 < len(chain_pts):
            add(pts[-1], chain_pts[ci + 1][0], "T", speed_t)
    add(chain_pts[-1][-1], home, "T", speed_t)

    if not result:
        raise ValueError("Zero-duration bundle")
    return result, t


def deposition_duration(candidate):
    return sum(t1 - t0 for t0, t1, _a, _b, mode in candidate if mode == "D")


def find_bundle_candidate(bundle, tasks, preferred, robots, tracks, available, config, max_retries):
    edge_ids = bundle_edge_ids(bundle, tasks)
    pref = preferred_robot_for_chain(edge_ids, preferred)
    feasible = []

    for robot in sorted(robots, key=lambda r: int(r["id"]) != pref):
        rid = int(robot["id"])
        try:
            local, duration = bundle_segments(robot, bundle, tasks, config)
        except ValueError:
            continue

        start = available[rid]
        found = False
        for _ in range(max_retries):
            candidate = shifted(local, start)
            conflict, next_start = candidate_conflict(robot, candidate, robots, tracks, config)
            if not conflict:
                found = True
                break
            if next_start is None or next_start <= start + 1e-9:
                break
            refined = refine_earliest_collision_free_start(
                robot, local, robots, tracks, config, start, next_start,
                coarse_step=1.0, refine_iters=14,
            )
            start = refined if refined is not None else next_start

        if found:
            feasible.append((
                start + duration,
                rid != pref,
                start,
                rid,
                candidate,
                deposition_duration(candidate),
            ))

    if not feasible:
        return None
    return min(feasible, key=lambda x: (x[0], x[1]))


def split_bundle(bundle):
    """Split by chain count.  If only one multi-edge chain remains, split its edges."""
    if len(bundle) > 1:
        mid = len(bundle) // 2
        return bundle[:mid], bundle[mid:]
    chain = bundle[0]
    if len(chain) <= 1:
        return None
    mid = len(chain) // 2
    return [chain[:mid]], [chain[mid:]]


def bundle_dep_time(bundle, tasks, config):
    speed = float(config["process"]["deposition_speed_mm_s"])
    total = 0.0
    for c in bundle:
        pts, _ = chain_points(c, tasks)
        total += sum(float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:]))
    return total / speed


def _run_bundle_schedule(bundles, chains, tasks, config, preferred, max_retries, label):
    robots = sorted(config["robots"], key=lambda r: int(r["id"]))
    bundles = sorted(bundles, key=lambda b: bundle_dep_time(b, tasks, config), reverse=True)

    print(f"{label} bundles: {len(bundles)}", flush=True)
    if bundles:
        sizes = np.asarray([len(b) for b in bundles], dtype=float)
        print(f"Bundle chain count: mean={sizes.mean():.2f}, max={int(sizes.max())}", flush=True)

    tracks = {int(r["id"]): [] for r in robots}
    available = {int(r["id"]): 0.0 for r in robots}
    dep_load = {int(r["id"]): 0.0 for r in robots}
    changes = parallel = split_count = scheduled_units = 0

    pending = [(b, 0) for b in bundles]
    while pending:
        bundle, depth = pending.pop(0)
        choice = find_bundle_candidate(
            bundle, tasks, preferred, robots, tracks, available, config, max_retries
        )
        if choice is None:
            pieces = split_bundle(bundle)
            if pieces is None:
                raise RuntimeError("V15 could not schedule even a single deposition edge safely")
            left, right = pieces
            split_count += 1
            pending.insert(0, (right, depth + 1))
            pending.insert(0, (left, depth + 1))
            continue

        finish, changed, start, rid, candidate, dep_dt = choice
        robot = next(r for r in robots if int(r["id"]) == rid)
        if start > available[rid] + 1e-10:
            h = xyz(robot, "home_xyz_mm")
            tracks[rid].append((available[rid], start, h.copy(), h.copy(), "W"))
        if any(start < available[oid] - 1e-9 for oid in available if oid != rid):
            parallel += 1
        tracks[rid].extend(candidate)
        available[rid] = candidate[-1][1]
        dep_load[rid] += dep_dt
        changes += int(changed)
        scheduled_units += 1
        if scheduled_units % 250 == 0:
            print(f"Scheduled bundle units: {scheduled_units} | pending={len(pending)}", flush=True)

    makespan = max(available.values())
    for robot in robots:
        rid = int(robot["id"])
        if available[rid] < makespan - 1e-10:
            h = xyz(robot, "home_xyz_mm")
            tracks[rid].append((available[rid], makespan, h.copy(), h.copy(), "W"))
    return tracks, makespan, changes, parallel, dep_load, split_count


def schedule_bundles(scenario, config, preferred, max_tasks, max_retries, join_tol):
    robots = sorted(config["robots"], key=lambda r: int(r["id"]))
    tasks = scenario["tasks"][:max_tasks] if max_tasks else scenario["tasks"]
    chains = chain_tasks(tasks, tol=join_tol)
    base_bundles = build_same_layer_bundles(chains, tasks, robots)

    print(f"Original centerline edges: {len(tasks)}", flush=True)
    print(f"Continuous chains: {len(chains)}", flush=True)
    print("V15 route: HOME -> chain -> direct T -> ... -> HOME", flush=True)
    print("V15 safety: V12 collision check + 0.05s refined boundary buffer", flush=True)

    first = _run_bundle_schedule(
        base_bundles, chains, tasks, config, preferred, max_retries,
        "Initial same-layer"
    )
    tracks, makespan, changes, parallel, dep_load, split_count = first

    active = {rid for rid, load in dep_load.items() if load > 1e-9}
    capable = geometrically_capable_robots(chains, tasks, robots)
    target_active = min(len(robots), len(capable))

    # Preserve the proven V14 schedule whenever it already uses all robots that
    # can geometrically contribute.  Only under-utilization triggers a rebuild.
    if len(active) < target_active:
        adaptive = adaptive_reach_bundles(base_bundles, tasks, robots)
        if len(adaptive) > len(base_bundles):
            print(
                f"Adaptive reach split triggered: active={sorted(active)} | "
                f"geometrically capable={sorted(capable)} | "
                f"bundles {len(base_bundles)} -> {len(adaptive)}",
                flush=True,
            )
            second = _run_bundle_schedule(
                adaptive, chains, tasks, config, preferred, max_retries,
                "Adaptive reach-region"
            )
            t2, m2, c2, p2, d2, s2 = second
            active2 = {rid for rid, load in d2.items() if load > 1e-9}
            # Prefer the adaptive schedule when it activates more robots; if robot
            # use ties, prefer lower makespan. Safety is audited after selection.
            if len(active2) > len(active) or (len(active2) == len(active) and m2 < makespan):
                tracks, makespan, changes, parallel, dep_load = t2, m2, c2, p2, d2
                split_count += s2
                chosen_bundle_count = len(adaptive)
                print("Selected adaptive reach-region schedule", flush=True)
            else:
                chosen_bundle_count = len(base_bundles)
                print("Kept initial V14-style schedule", flush=True)
        else:
            chosen_bundle_count = len(base_bundles)
            print("Adaptive split found no geometry-derived reach boundary; kept initial schedule", flush=True)
    else:
        chosen_bundle_count = len(base_bundles)
        print("Initial schedule already uses all geometrically capable robots; no adaptive split", flush=True)

    return tracks, makespan, changes, parallel, len(chains), chosen_bundle_count, dep_load, split_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=100)
    ap.add_argument("--join-tol", type=float, default=1e-4)
    ap.add_argument("--out", default="outputs/trajectory_CENTERLINE_V15_ADAPTIVE.csv")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    scenario = build_scenario(args.stl, args.config)
    env = WAAMBaselineEnv(scenario)
    model = PPO.load(args.model)
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    limit = args.max_tasks if args.max_tasks > 0 else None
    result = schedule_bundles(
        scenario, config, env.assignment, limit, args.max_retries, args.join_tol
    )
    tracks, makespan, changes, parallel, chain_count, bundle_count, dep_load, split_count = result

    print("Tasks:", limit or len(scenario["tasks"]), flush=True)
    print("Chains:", chain_count, flush=True)
    print("Initial bundles:", bundle_count, flush=True)
    print("Fallback bundle splits:", split_count, flush=True)
    print("Makespan:", makespan, flush=True)
    print(
        "Deposition load:",
        " | ".join(f"R{rid}={dep_load[rid]:.2f}s" for rid in sorted(dep_load)),
        flush=True,
    )
    print("PPO preference changes:", changes, flush=True)
    print("Parallel bundle launches:", parallel, flush=True)

    print("Running complete sampled collision audit...", flush=True)
    count, first, margin = audit(tracks, config, makespan)
    print("Collision samples:", count, flush=True)
    print("First collision:", first, flush=True)
    print("Min arm margin:", margin, flush=True)
    if count:
        raise RuntimeError("Collision audit failed; no CSV exported")

    out = write_validator_csv(tracks, args.out)
    print("Validator-format V15 CSV:", out, flush=True)
    print("Run the professor Validator; this script does not claim official PASS.", flush=True)


if __name__ == "__main__":
    main()
