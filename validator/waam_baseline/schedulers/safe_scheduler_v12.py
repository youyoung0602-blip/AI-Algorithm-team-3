"""V12: makespan-first centerline scheduler with a mild load-balance penalty.

Based on the validated V4 logic:
- Same centerline geometry/chaining
- Same collision-aware candidate search
- Same validator CSV interval-mode semantics

V7 change:
- Candidate selection primarily minimizes projected global makespan.
- A small deposition-load imbalance penalty is used only as a secondary influence.
- PPO chain preference remains a tie-breaker / small preference, not the main objective.

Requires beside this file:
- safe_scheduler_v2.py
- export_and_check_v2.py
- task_generator_v2.py
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

from algorithm.environment.waam_env import WAAMBaselineEnv
from task_generator_v2 import build_scenario
from safe_scheduler_v2 import xyz, shifted, candidate_conflict
from export_and_check_v2 import audit


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


def point_key(p, tol):
    p = np.asarray(p, dtype=float)
    return tuple(np.rint(p / tol).astype(np.int64).tolist())


def chain_tasks(tasks, tol=1e-4):
    n = len(tasks)
    starts = [np.asarray(t["start_xyz_mm"], dtype=float) for t in tasks]
    ends = [np.asarray(t["end_xyz_mm"], dtype=float) for t in tasks]

    adjacency = {}
    for i in range(n):
        adjacency.setdefault(point_key(starts[i], tol), []).append((i, 0))
        adjacency.setdefault(point_key(ends[i], tol), []).append((i, 1))

    unused = set(range(n))
    chains = []

    seeds = []
    for i in range(n):
        ks, ke = point_key(starts[i], tol), point_key(ends[i], tol)
        if len(adjacency.get(ks, [])) == 1 or len(adjacency.get(ke, [])) == 1:
            seeds.append(i)
    seeds.extend(range(n))

    def grow(seed):
        if seed not in unused:
            return None

        ks, ke = point_key(starts[seed], tol), point_key(ends[seed], tol)
        if len(adjacency.get(ke, [])) == 1 and len(adjacency.get(ks, [])) != 1:
            edges = [(seed, True)]
            front = ends[seed].copy()
            back = starts[seed].copy()
        else:
            edges = [(seed, False)]
            front = starts[seed].copy()
            back = ends[seed].copy()
        unused.remove(seed)

        while True:
            k = point_key(back, tol)
            candidates = [(j, side) for j, side in adjacency.get(k, []) if j in unused]
            if not candidates:
                break
            j, side = min(candidates, key=lambda x: x[0])
            rev = (side == 1)
            edges.append((j, rev))
            unused.remove(j)
            back = (starts[j] if rev else ends[j]).copy()

        while True:
            k = point_key(front, tol)
            candidates = [(j, side) for j, side in adjacency.get(k, []) if j in unused]
            if not candidates:
                break
            j, side = min(candidates, key=lambda x: x[0])
            rev = (side == 0)
            edges.insert(0, (j, rev))
            unused.remove(j)
            front = (ends[j] if rev else starts[j]).copy()

        return edges

    for seed in seeds:
        c = grow(seed)
        if c:
            chains.append(c)
        if not unused:
            break

    if unused:
        raise RuntimeError(f"Internal chaining error: {len(unused)} edges unused")
    return chains


def chain_points(chain, tasks):
    pts, ids = [], []
    for pos, (idx, rev) in enumerate(chain):
        t = tasks[idx]
        a = np.asarray(t["end_xyz_mm"] if rev else t["start_xyz_mm"], dtype=float)
        b = np.asarray(t["start_xyz_mm"] if rev else t["end_xyz_mm"], dtype=float)
        if pos == 0:
            pts.append(a.copy())
        elif np.linalg.norm(pts[-1] - a) > 1e-3:
            raise RuntimeError("Chain contains a discontinuity")
        pts.append(b.copy())
        ids.append(idx)
    return pts, ids


def chain_segments(robot, chain, tasks, config):
    points, edge_ids = chain_points(chain, tasks)
    home = xyz(robot, "home_xyz_mm")
    base = xyz(robot, "base_xyz_mm")
    reach = float(robot["xy_reach_radius_mm"])
    speed_t = float(config["process"]["travel_speed_mm_s"])
    speed_d = float(config["process"]["deposition_speed_mm_s"])

    if min(speed_t, speed_d) <= 0:
        raise ValueError("Speeds must be positive")

    for p in [home] + points:
        if np.linalg.norm(p[:2] - base[:2]) > reach + 1e-7:
            raise ValueError("XY reach exceeded")

    result, t = [], 0.0

    def add(a, b, mode, speed):
        nonlocal t
        dt = float(np.linalg.norm(b - a)) / speed
        if dt > 1e-10:
            result.append((t, t + dt, a.copy(), b.copy(), mode))
            t += dt

    add(home, points[0], "T", speed_t)
    for a, b in zip(points[:-1], points[1:]):
        add(a, b, "D", speed_d)
    add(points[-1], home, "T", speed_t)

    if not result:
        raise ValueError("Zero-duration chain")
    return result, t, edge_ids


def preferred_robot_for_chain(edge_ids, preferred):
    votes = {}
    for i in edge_ids:
        rid = int(preferred[i])
        votes[rid] = votes.get(rid, 0) + 1
    return min(votes, key=lambda rid: (-votes[rid], rid))


def deposition_duration(candidate):
    return sum((t1 - t0) for t0, t1, _a, _b, mode in candidate if mode == "D")



def order_chains_longest_first(chains, tasks, config):
    """Schedule longer deposition chains first.

    The 36-chain geometry is unchanged.  Only chain processing order changes.
    Long chains are placed earlier so the scheduler has more opportunity to
    overlap them across robots; short chains fill the remaining gaps later.
    """
    speed_d = float(config["process"]["deposition_speed_mm_s"])
    ranked = []
    for original_pos, chain in enumerate(chains):
        pts, _ = chain_points(chain, tasks)
        dep_len = sum(
            float(np.linalg.norm(b - a))
            for a, b in zip(pts[:-1], pts[1:])
        )
        dep_time = dep_len / speed_d
        ranked.append((dep_time, original_pos, chain))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in ranked]



def refine_earliest_collision_free_start(
    robot, local, robots, tracks, config, start, suggested_start,
    coarse_step=1.0, refine_iters=14
):
    """Find an earlier collision-free launch than a coarse conflict jump.

    candidate_conflict() may return a conservative next feasible boundary.
    This helper searches the interval [start, suggested_start] and returns the
    earliest collision-free start found. It never accepts a colliding candidate.
    """
    if suggested_start is None or suggested_start <= start + 1e-9:
        return None

    # Coarse forward scan. Keep it bounded: chains are only 36 in this case.
    t = start + coarse_step
    last_bad = start
    first_good = None

    while t < suggested_start - 1e-9:
        cand = shifted(local, t)
        conflict, _ = candidate_conflict(robot, cand, robots, tracks, config)
        if not conflict:
            first_good = t
            break
        last_bad = t
        t += coarse_step

    if first_good is None:
        cand = shifted(local, suggested_start)
        conflict, _ = candidate_conflict(robot, cand, robots, tracks, config)
        if conflict:
            return None
        first_good = suggested_start

    # Binary refinement between last known collision and first known safe time.
    lo, hi = last_bad, first_good
    for _ in range(refine_iters):
        mid = 0.5 * (lo + hi)
        cand = shifted(local, mid)
        conflict, _ = candidate_conflict(robot, cand, robots, tracks, config)
        if conflict:
            lo = mid
        else:
            hi = mid

    # V10 safety buffer:
    # V9 stopped almost exactly on the collision boundary and the final sampled
    # audit found a tiny negative arm margin (~-0.0025 mm). Move slightly beyond
    # the refined boundary, then verify again before accepting it.
    safety_dt = 0.05
    safe_t = hi + safety_dt

    # If numerical/sampling differences still report a conflict, move forward
    # in small increments until candidate_conflict confirms safety.
    for _ in range(20):
        cand = shifted(local, safe_t)
        conflict, _ = candidate_conflict(robot, cand, robots, tracks, config)
        if not conflict:
            return safe_t
        safe_t += safety_dt

    # Fall back to the already collision-free binary-search boundary if the
    # extra-buffer verification unexpectedly fails.
    return hi



def split_chain_in_half(chain):
    """Split one chain by original edge sequence; geometry itself is unchanged."""
    if len(chain) <= 1:
        return None
    mid = len(chain) // 2
    if mid <= 0 or mid >= len(chain):
        return None
    return list(chain[:mid]), list(chain[mid:])


def find_chain_candidate(
    chain, tasks, preferred, robots, tracks, available, config, max_retries
):
    """Return the best feasible robot candidate for one chain, or None."""
    _, edge_ids = chain_points(chain, tasks)
    pref = preferred_robot_for_chain(edge_ids, preferred)
    feasible = []

    for robot in sorted(robots, key=lambda r: int(r["id"]) != pref):
        rid = int(robot["id"])
        try:
            local, duration, _ = chain_segments(robot, chain, tasks, config)
        except ValueError:
            continue

        start = available[rid]
        found = False
        for _ in range(max_retries):
            candidate = shifted(local, start)
            conflict, next_start = candidate_conflict(
                robot, candidate, robots, tracks, config
            )
            if not conflict:
                found = True
                break
            if next_start is None or next_start <= start + 1e-9:
                break

            refined = refine_earliest_collision_free_start(
                robot, local, robots, tracks, config,
                start, next_start,
                coarse_step=1.0,
                refine_iters=14,
            )
            start = refined if refined is not None else next_start

        if found:
            dep_dt = deposition_duration(candidate)
            feasible.append((
                start + duration,
                rid != pref,
                start,
                rid,
                candidate,
                dep_dt,
            ))

    if not feasible:
        return None
    return min(feasible, key=lambda x: (x[0], x[1]))


def expand_infeasible_chains(
    chains, tasks, preferred, robots, tracks, available, config,
    max_retries, max_split_depth=8
):
    """Yield schedulable chains; recursively split only a chain that is blocked.

    This is a fallback for test cases where a whole continuous chain cannot be
    assigned collision-free. Original deposition edges are never altered.
    """
    pending = [(list(c), 0) for c in chains]
    out = []

    while pending:
        chain, depth = pending.pop(0)
        candidate = find_chain_candidate(
            chain, tasks, preferred, robots, tracks, available,
            config, max_retries
        )

        if candidate is not None:
            out.append((chain, candidate, depth))
            # Reserve immediately so later feasibility checks see this schedule.
            _, _, start, rid, segs, _ = candidate
            robot = next(r for r in robots if int(r["id"]) == rid)
            if start > available[rid] + 1e-10:
                h = xyz(robot, "home_xyz_mm")
                tracks[rid].append(
                    (available[rid], start, h.copy(), h.copy(), "W")
                )
            tracks[rid].extend(segs)
            available[rid] = segs[-1][1]
            continue

        pieces = split_chain_in_half(chain)
        if pieces is None or depth >= max_split_depth:
            return None

        # Preserve deposition order inside the original chain.
        left, right = pieces
        pending.insert(0, (right, depth + 1))
        pending.insert(0, (left, depth + 1))

    return out


def schedule_one_order(
    scenario, config, preferred, max_tasks, max_retries, join_tol, chains
):
    """V12: V10 collision logic + recursive split fallback for blocked chains."""
    robots = sorted(config["robots"], key=lambda r: int(r["id"]))
    tracks = {int(r["id"]): [] for r in robots}
    available = {int(r["id"]): 0.0 for r in robots}
    dep_load = {int(r["id"]): 0.0 for r in robots}

    tasks = scenario["tasks"][:max_tasks] if max_tasks else scenario["tasks"]
    changes = 0
    parallel = 0
    split_count = 0

    # Process one requested chain at a time. If blocked, recursively split only
    # that chain. This keeps Test 01 behavior unchanged unless fallback is needed.
    for original_chain in chains:
        queue = [(list(original_chain), 0)]

        while queue:
            chain, depth = queue.pop(0)
            choice = find_chain_candidate(
                chain, tasks, preferred, robots, tracks, available,
                config, max_retries
            )

            if choice is None:
                pieces = split_chain_in_half(chain)
                if pieces is None or depth >= 8:
                    return None
                left, right = pieces
                split_count += 1
                queue.insert(0, (right, depth + 1))
                queue.insert(0, (left, depth + 1))
                continue

            finish, changed, start, rid, candidate, dep_dt = choice
            robot = next(r for r in robots if int(r["id"]) == rid)

            if start > available[rid] + 1e-10:
                h = xyz(robot, "home_xyz_mm")
                tracks[rid].append(
                    (available[rid], start, h.copy(), h.copy(), "W")
                )

            if any(
                start < available[oid] - 1e-9
                for oid in available if oid != rid
            ):
                parallel += 1

            tracks[rid].extend(candidate)
            available[rid] = candidate[-1][1]
            dep_load[rid] += dep_dt
            changes += int(changed)

    makespan = max(available.values())
    for robot in robots:
        rid = int(robot["id"])
        if available[rid] < makespan - 1e-10:
            h = xyz(robot, "home_xyz_mm")
            tracks[rid].append(
                (available[rid], makespan, h.copy(), h.copy(), "W")
            )

    return tracks, makespan, changes, parallel, dep_load, split_count


def chain_dep_time(chain, tasks, config):
    pts, _ = chain_points(chain, tasks)
    length = sum(
        float(np.linalg.norm(b - a))
        for a, b in zip(pts[:-1], pts[1:])
    )
    return length / float(config["process"]["deposition_speed_mm_s"])


def make_candidate_orders(chains, tasks, config, trials, seed):
    """Create deterministic order portfolio.

    Candidate 0 is the ORIGINAL V4 chain order, so V7 always keeps the
    validated V4 ordering as a baseline. Other candidates explore alternatives.
    """
    orders = []
    seen = set()

    def add(name, order):
        key = tuple(id(c) for c in order)
        if key not in seen:
            seen.add(key)
            orders.append((name, list(order)))

    add("original_V4", chains)

    by_long = sorted(
        chains, key=lambda c: chain_dep_time(c, tasks, config), reverse=True
    )
    add("longest_first", by_long)
    add("shortest_first", list(reversed(by_long)))

    # Interleave long/short chains.
    interleaved = []
    lo, hi = 0, len(by_long) - 1
    while lo <= hi:
        interleaved.append(by_long[lo])
        lo += 1
        if lo <= hi:
            interleaved.append(by_long[hi])
            hi -= 1
    add("long_short_interleave", interleaved)

    # Deterministic random permutations.
    rng = random.Random(seed)
    for i in range(max(0, trials)):
        order = list(chains)
        rng.shuffle(order)
        add(f"random_{i+1:03d}", order)

    return orders


def schedule_chains(
    scenario, config, preferred, max_tasks, max_retries, join_tol,
    trials=40, seed=42
):
    """Search several chain orders and keep the lowest collision-aware makespan."""
    tasks = scenario["tasks"][:max_tasks] if max_tasks else scenario["tasks"]
    base_chains = chain_tasks(tasks, tol=join_tol)
    # V11: generalize across test cases.
    # Keep original order as the first baseline, then try deterministic
    # alternatives and seeded random permutations. Infeasible orders are
    # skipped rather than terminating the whole test case.
    edge_count = len(tasks)
    chain_count = len(base_chains)

    # Geometry-aware search budget.
    # Large edge counts make each collision-aware schedule expensive, so use
    # fewer random order trials. Fragmented geometry gets a few extra trials.
    if edge_count >= 20000:
        adaptive_trials = min(trials, 4)
        feasible_patience = 2
    elif edge_count >= 12000:
        adaptive_trials = min(trials, 8)
        feasible_patience = 3
    else:
        adaptive_trials = min(trials, 12)
        feasible_patience = 4

    if chain_count >= 50:
        adaptive_trials = min(trials, adaptive_trials + 4)

    orders = make_candidate_orders(
        base_chains, tasks, config, trials=adaptive_trials, seed=seed
    )

    print(f"Original centerline edges: {edge_count}", flush=True)
    print(f"Continuous chains: {chain_count}", flush=True)
    print(
        f"V12 candidate chain orders: {len(orders)} "
        f"(adaptive trials={adaptive_trials})",
        flush=True,
    )
    print("V12 collision handling: refined earliest-safe start + 0.05s safety buffer", flush=True)
    print("V12 behavior: geometry-aware search + early stopping", flush=True)

    best = None
    best_name = None
    feasible_seen = 0
    no_improve_after_best = 0

    for i, (name, order) in enumerate(orders, start=1):
        result = schedule_one_order(
            scenario, config, preferred, max_tasks,
            max_retries, join_tol, order
        )
        if result is None:
            print(f"[{i}/{len(orders)}] {name}: infeasible", flush=True)
            continue

        tracks, makespan, changes, parallel, dep_load, split_count = result
        feasible_seen += 1
        print(
            f"[{i}/{len(orders)}] {name}: "
            f"makespan={makespan:.3f}s, parallel={parallel}",
            flush=True,
        )

        if best is None or makespan < best[1] - 1e-9:
            best = (tracks, makespan, changes, parallel, dep_load, split_count)
            best_name = name
            no_improve_after_best = 0
            print(
                f"  -> NEW BEST: {best_name} / {makespan:.3f}s",
                flush=True,
            )
        else:
            no_improve_after_best += 1

        # Stop once several feasible alternatives fail to improve the current
        # best. This avoids exhausting dozens of expensive full collision runs.
        if (
            best is not None
            and feasible_seen >= feasible_patience
            and no_improve_after_best >= feasible_patience - 1
        ):
            print(
                f"Early stop: {feasible_seen} feasible candidates checked; "
                f"best={best_name} / {best[1]:.3f}s",
                flush=True,
            )
            break

    if best is None:
        raise RuntimeError("V12 could not find a feasible schedule even after chain splitting.")

    tracks, makespan, changes, parallel, dep_load, split_count = best
    print(f"V12 best order: {best_name}", flush=True)
    return (
        tracks, makespan, changes, parallel,
        len(base_chains), dep_load, best_name, split_count
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-retries", type=int, default=100)
    ap.add_argument("--join-tol", type=float, default=1e-4)
    ap.add_argument(
        "--trials", type=int, default=60,
        help="Maximum random chain-order trials; V12 automatically reduces this for large geometries (default: 60)"
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for chain-order search"
    )
    ap.add_argument("--out", default="outputs/trajectory_CENTERLINE_V12.csv")
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

    tracks, makespan, changes, parallel, chain_count, dep_load, best_name, split_count = schedule_chains(
        scenario, config, env.assignment, limit,
        args.max_retries, args.join_tol,
        args.trials, args.seed,
    )

    print("Tasks:", limit or len(scenario["tasks"]), flush=True)
    print("Chains:", chain_count, flush=True)
    print("Best chain order:", best_name, flush=True)
    print("Fallback chain splits:", split_count, flush=True)
    print("Makespan:", makespan, flush=True)
    print(
        "Deposition load:",
        " | ".join(f"R{rid}={dep_load[rid]:.2f}s" for rid in sorted(dep_load)),
        flush=True,
    )
    print("Chain PPO preference changes:", changes, flush=True)
    print("Parallel chain launches:", parallel, flush=True)

    print("Running complete sampled collision audit...", flush=True)
    count, first, margin = audit(tracks, config, makespan)
    print("Collision samples:", count, flush=True)
    print("First collision:", first, flush=True)
    print("Min arm margin:", margin, flush=True)

    if count:
        raise RuntimeError("Collision audit failed; no CSV exported")

    out = write_validator_csv(tracks, args.out)
    print("Validator-format V12 CSV:", out, flush=True)


if __name__ == "__main__":
    main()
