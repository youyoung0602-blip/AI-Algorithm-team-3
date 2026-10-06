"""Diagnostic PPO trajectory export and vectorized collision audit.
NOT a safe scheduler or submission-ready WAAM trajectory.
Run from validator/waam_sb3_baseline after installing waam_validator.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import yaml
from stable_baselines3 import PPO
from algorithm.environment.waam_env import WAAMBaselineEnv
from algorithm.preprocessing.task_generator import build_scenario
from waam_validator.collision import check_arm_envelope_xy_batch


def make_schedule(scenario, assignments, config):
    robots = sorted(config['robots'], key=lambda r: int(r['id']))
    speed_t = float(config['process']['travel_speed_mm_s'])
    speed_d = float(config['process']['deposition_speed_mm_s'])
    tracks = {int(r['id']): [] for r in robots}
    pos = {int(r['id']): np.asarray(r['home_xyz_mm'], dtype=float) for r in robots}
    clocks = {int(r['id']): 0.0 for r in robots}

    def add(rid, destination, mode, speed):
        a, b = pos[rid].copy(), np.asarray(destination, dtype=float)
        duration = float(np.linalg.norm(b - a) / speed)
        if duration > 1e-10:
            t = clocks[rid]
            tracks[rid].append((t, t + duration, a, b, mode))
            clocks[rid] = t + duration
        pos[rid] = b

    for task, assigned in zip(scenario['tasks'], assignments, strict=True):
        rid = int(assigned)
        if rid not in tracks:
            raise ValueError(f'Invalid assignment: {rid}')
        start = np.asarray(task['start_xyz_mm'], dtype=float)
        end = np.asarray(task['end_xyz_mm'], dtype=float)
        # Direct T: matches current PPO travel-distance model. NOT certified safe.
        add(rid, start, 'T', speed_t)
        add(rid, end, 'D', speed_d)

    makespan = max(clocks.values())
    for r in robots:
        rid = int(r['id'])
        if clocks[rid] < makespan:
            p = pos[rid].copy()
            tracks[rid].append((clocks[rid], makespan, p, p, 'W'))
    return tracks, makespan


def positions_at_times(track, times, home):
    """Vectorized piecewise-linear TCP interpolation, including initial waiting."""
    result = np.broadcast_to(np.asarray(home, dtype=float), (len(times), 3)).copy()
    if not track:
        return result
    ends = np.asarray([s[1] for s in track], dtype=float)
    starts = np.asarray([s[0] for s in track], dtype=float)
    a = np.asarray([s[2] for s in track], dtype=float)
    b = np.asarray([s[3] for s in track], dtype=float)
    idx = np.searchsorted(ends, times, side='left')
    valid = idx < len(track)
    ii = idx[valid]
    frac = np.clip((times[valid] - starts[ii]) / (ends[ii] - starts[ii]), 0, 1)
    result[valid] = a[ii] + frac[:, None] * (b[ii] - a[ii])
    result[~valid] = b[-1]
    return result


def audit(tracks, config, makespan):
    robots = sorted(config['robots'], key=lambda r: int(r['id']))
    cc, sim = config['collision'], config['simulation']
    dt = float(sim['max_time_step_s'])
    batch_size = int(sim.get('batch_size', 10000))
    if dt <= 0 or batch_size <= 0:
        raise ValueError('Invalid simulation sampling configuration')
    n = int(np.ceil(makespan / dt))
    count, first, worst_margin = 0, None, float('inf')
    for offset in range(0, n + 1, batch_size):
        times = np.minimum(np.arange(offset, min(offset + batch_size, n + 1)) * dt, makespan)
        xyz = [positions_at_times(tracks[int(r['id'])], times, r['home_xyz_mm']) for r in robots]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            a, b = robots[i], robots[j]
            arm = check_arm_envelope_xy_batch(
                np.asarray(a['base_xyz_mm'][:2], dtype=float), xyz[i][:, :2],
                float(a['arm_envelope_radius_mm']),
                np.asarray(b['base_xyz_mm'][:2], dtype=float), xyz[j][:, :2],
                float(b['arm_envelope_radius_mm']),
                float(cc['arm_clearance_mm']), float(cc['geometry_epsilon_mm']),
                bool(cc['touching_is_collision']),
            )
            distance = np.linalg.norm(xyz[i][:, :2] - xyz[j][:, :2], axis=1)
            required_tcp = float(a['tcp_radius_mm']) + float(b['tcp_radius_mm'])
            if cc['touching_is_collision']:
                tcp_collision = distance <= required_tcp
            else:
                tcp_collision = distance < required_tcp
            collision = ((arm.collision if cc['check_arm_envelope'] else False) |
                         (tcp_collision if cc['check_tcp_radius'] else False))
            worst_margin = min(worst_margin, float(np.min(arm.safety_margin_mm)))
            count += int(np.count_nonzero(collision))
            if np.any(collision):
                k = int(np.flatnonzero(collision)[0])
                event = (float(times[k]), int(a['id']), int(b['id']),
                         float(arm.safety_margin_mm[k]))
                if first is None or event[0] < first[0]:
                    first = event
        if offset == 0 or offset // batch_size % 10 == 0:
            print(f'Collision audit: {min(offset + len(times), n + 1):,}/{n + 1:,} samples', flush=True)
    return count, first, worst_margin


def write_csv(tracks, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['robot_id', 'time_s', 'x_mm', 'y_mm', 'z_mm', 'mode'])
        for rid, track in tracks.items():
            if not track:
                continue
            t0, _, start, _, _ = track[0]
            w.writerow([rid, f'{t0:.8f}', *[f'{v:.8f}' for v in start], 'W'])
            for _, t1, _, end, mode in track:
                w.writerow([rid, f'{t1:.8f}', *[f'{v:.8f}' for v in end], mode])
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stl', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', default='outputs/trajectory_DIAGNOSTIC_v2.csv')
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    scenario = build_scenario(args.stl, args.config)
    env = WAAMBaselineEnv(scenario)
    model = PPO.load(args.model)
    obs, _ = env.reset(seed=42)
    while True:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    if not np.all(env.assignment > 0):
        raise RuntimeError('Not all tasks were assigned')
    tracks, makespan = make_schedule(scenario, env.assignment, config)
    print('PPO internal makespan:', round(float(np.max(env.robot_time)), 3), 's', flush=True)
    print('Direct-travel trajectory makespan:', round(makespan, 3), 's', flush=True)
    output = write_csv(tracks, args.out)
    print('Diagnostic CSV:', output, flush=True)
    print('Auditing using professor\'s vectorized collision geometry...', flush=True)
    count, first, margin = audit(tracks, config, makespan)
    print('Colliding pair-samples:', count)
    print('First collision (time, robot A, robot B, arm margin):', first)
    print('Minimum arm safety margin:', round(margin, 3), 'mm')
    if count:
        print('FAIL: Collision detected. Do NOT submit this CSV.')
    else:
        print('No sampled collision detected. Full Validator is STILL required.')
    print('DIAGNOSTIC ONLY: no safe scheduler, contour-only deposition, and preliminary CSV encoding.')


if __name__ == '__main__':
    main()
