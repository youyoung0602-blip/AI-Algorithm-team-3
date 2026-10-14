"""V17: reach-feasible chain partitioning before audited neighborhood search.

Keeps V15 as a fallback. Tests bundle order permutations and reach-aware splits,
reusing V15's collision-aware scheduler. No PPO retraining required.
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import yaml
from stable_baselines3 import PPO

from algorithm.environment.waam_env import WAAMBaselineEnv
from task_generator_v2 import build_scenario
from export_and_check_v2 import audit
from safe_scheduler_v12 import chain_tasks
from safe_scheduler_v15_adaptive_bundles import (
    build_same_layer_bundles, adaptive_reach_bundles, bundle_dep_time,
    bundle_edge_ids, _run_bundle_schedule, schedule_bundles as v15_schedule,
    write_validator_csv,
)


def split_chains_by_reach(chains, tasks, robots):
    """Cut before the shared feasible robot set becomes empty.

    Keeps every deposition edge, direction, coordinate and layer unchanged.
    Endpoints inside a robot's XY reach disk imply the whole straight edge
    is inside it, because a disk is convex. Does not use test identifiers.
    """
    import numpy as np
    output = []
    signatures = {}
    for chain in chains:
        current = []
        shared = set()
        for edge in chain:
            task_id, reverse = edge
            if task_id not in signatures:
                task = tasks[task_id]
                points = np.asarray([task['start_xyz_mm'], task['end_xyz_mm']], float)[:, :2]
                signatures[task_id] = {
                    int(r['id']) for r in robots
                    if np.all(np.linalg.norm(points - np.asarray(r['base_xyz_mm'], float)[:2], axis=1)
                              <= float(r['xy_reach_radius_mm']) + 1e-7)
                }
            feasible = signatures[task_id]
            if not feasible:
                raise ValueError(f'Deposition edge {task_id} is outside every robot XY reach')
            common = shared & feasible if current else set(feasible)
            if current and not common:
                output.append(current)
                current = []
                common = set(feasible)
            current.append(edge)
            shared = common
        if current:
            output.append(current)
    before = [edge for chain in chains for edge in chain]
    after = [edge for chain in output for edge in chain]
    if before != after:
        raise RuntimeError('Reach splitting changed deposition edge coverage or direction')
    return output


def schedule_v17(scenario, config, preferred, max_tasks=None, max_retries=100,
                 join_tol=1e-4, iterations=20, seed=42, max_bundles=300,
                 audit_each=True):
    import time
    tasks = scenario['tasks'][:max_tasks] if max_tasks else scenario['tasks']
    v15 = v15_schedule(scenario, config, preferred, max_tasks, max_retries, join_tol)
    best_tracks, best_makespan = v15[0], v15[1]
    base_count = v15[5]
    if audit_each:
        count, _, _ = audit(best_tracks, config, best_makespan)
        if count:
            raise RuntimeError('V15 baseline failed collision audit')
    print(f'V15 baseline makespan: {best_makespan:.4f}s', flush=True)
    print('Baseline deposition seconds per robot:', v15[6], flush=True)

    original_chains = chain_tasks(tasks, tol=join_tol)
    chains = split_chains_by_reach(original_chains, tasks, config['robots'])
    print(f'V17 reach chains: {len(original_chains)} -> {len(chains)}; edges preserved: {len(tasks)}', flush=True)
    base = build_same_layer_bundles(chains, tasks, config['robots'])
    # Preserve the audited fallback if optional variant construction fails.
    variants = [base]
    try:
        adaptive = adaptive_reach_bundles(base, tasks, config['robots'])
        if adaptive != base:
            variants.append(adaptive)
    except (RuntimeError, ValueError) as exc:
        print(f'Adaptive variant skipped; audited fallback retained: {exc}', flush=True)
    print('V17 variant bundle counts:', [len(v) for v in variants], flush=True)
    rng = random.Random(seed)
    tested = accepted = 0
    for trial in range(max(0, iterations)):
        bundles = list(variants[trial % len(variants)])
        if not bundles or len(bundles) > max_bundles:
            print(f'V17 trial {trial+1}/{iterations} skipped: bundle limit ({len(bundles)})', flush=True)
            continue
        order = list(range(len(bundles)))
        if trial == 0:
            order.sort(key=lambda i: bundle_dep_time(bundles[i], tasks, config))
        elif trial == 1:
            order.sort(key=lambda i: bundle_dep_time(bundles[i], tasks, config), reverse=True)
        else:
            order.sort(key=lambda i: bundle_dep_time(bundles[i], tasks, config), reverse=True)
            for _ in range(1 + trial % 5):
                a, b = rng.sample(range(len(order)), 2) if len(order) >= 2 else (0, 0)
                order[a], order[b] = order[b], order[a]
        candidate_bundles = [bundles[i] for i in order]
        started = time.monotonic()
        print(f'V17 trial {trial+1}/{iterations} started: bundles={len(bundles)}', flush=True)
        try:
            result = run_ordered(candidate_bundles, tasks, config, preferred, max_retries)
        except (RuntimeError, ValueError) as exc:
            print(f'V17 trial {trial+1}/{iterations} rejected: {exc}', flush=True)
            continue
        tested += 1
        tracks, makespan = result[0], result[1]
        print(f'V17 trial {trial+1}/{iterations} finished: makespan={makespan:.4f}s; compute={time.monotonic()-started:.1f}s', flush=True)
        if makespan >= best_makespan - 1e-7:
            continue
        if audit_each:
            collisions, _, _ = audit(tracks, config, makespan)
            if collisions:
                print(f'V17 trial {trial+1}/{iterations} rejected by collision audit', flush=True)
                continue
        best_tracks, best_makespan = tracks, makespan
        accepted += 1
        print(f'V17 improvement trial={trial}: {makespan:.4f}s', flush=True)
    return best_tracks, best_makespan, v15[1], tested, accepted, base_count


def run_ordered(bundles, tasks, config, preferred, max_retries):
    """V15 collision-safe placement, preserving the supplied bundle order."""
    from safe_scheduler_v15_adaptive_bundles import (
        find_bundle_candidate, split_bundle, xyz,
    )
    import numpy as np
    robots = sorted(config['robots'], key=lambda r: int(r['id']))
    tracks = {int(r['id']): [] for r in robots}
    available = {int(r['id']): 0.0 for r in robots}
    pending = list(bundles)
    while pending:
        bundle = pending.pop(0)
        choice = find_bundle_candidate(bundle, tasks, preferred, robots, tracks,
                                       available, config, max_retries)
        if choice is None:
            parts = split_bundle(bundle)
            if parts is None:
                raise RuntimeError('Unschedulable deposition edge')
            pending[:0] = list(parts)
            continue
        _, _, start, rid, candidate, _ = choice
        robot = next(r for r in robots if int(r['id']) == rid)
        if start > available[rid] + 1e-10:
            h = xyz(robot, 'home_xyz_mm')
            tracks[rid].append((available[rid], start, h.copy(), h.copy(), 'W'))
        tracks[rid].extend(candidate)
        available[rid] = candidate[-1][1]
    makespan = max(available.values())
    for robot in robots:
        rid = int(robot['id'])
        if available[rid] < makespan - 1e-10:
            h = xyz(robot, 'home_xyz_mm')
            tracks[rid].append((available[rid], makespan, h.copy(), h.copy(), 'W'))
    return tracks, makespan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stl', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='outputs/trajectory_CENTERLINE_V17.csv')
    ap.add_argument('--max-tasks', type=int, default=0)
    ap.add_argument('--max-retries', type=int, default=100)
    ap.add_argument('--join-tol', type=float, default=1e-4)
    ap.add_argument('--iterations', type=int, default=20)
    ap.add_argument('--max-bundles', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    scenario = build_scenario(args.stl, args.config)
    env = WAAMBaselineEnv(scenario)
    model = PPO.load(args.model)
    obs, _ = env.reset(seed=args.seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            break
    tracks, makespan, baseline, tested, accepted, _ = schedule_v17(
        scenario, config, env.assignment, args.max_tasks or None,
        args.max_retries, args.join_tol, args.iterations, args.seed,
        args.max_bundles)
    collisions, first, margin = audit(tracks, config, makespan)
    if collisions:
        raise RuntimeError(f'Collision audit failed: {collisions}, first={first}')
    out = write_validator_csv(tracks, args.out)
    print(f'V15 makespan: {baseline:.4f}s')
    print(f'V17 makespan: {makespan:.4f}s')
    print(f'Improvement: {(baseline-makespan)/baseline*100 if baseline else 0:.2f}%')
    print(f'Tested candidates: {tested}; accepted improvements: {accepted}')
    print(f'Collision samples: {collisions}; minimum arm margin: {margin}')
    print(f'CSV: {out}')
    print('Run the official Validator; internal audit does not imply PASS.')


if __name__ == '__main__':
    main()
