"""Parallel, collision-aware diagnostic scheduler for 3-robot WAAM.

Conservative policy: every task starts at the robot's HOME and returns HOME.
Other robots may execute tasks simultaneously only if sampled arm/TCP checks pass.
The existing PPO assignment is preferred; alternative robots are attempted.
Fails closed if no candidate is feasible; never labels a colliding CSV safe.
Not a final WAAM submission: contour-only, simplified travel, no infill/thermal/
layer precedence certification. Keep the professor's Validator unchanged.
"""
import argparse
import math
import numpy as np
import yaml
from stable_baselines3 import PPO
from algorithm.environment.waam_env import WAAMBaselineEnv
from algorithm.preprocessing.task_generator import build_scenario
from waam_validator.collision import check_arm_envelope_xy_batch
from export_and_check_v2 import audit, write_csv, positions_at_times


def xyz(r, field):
    return np.asarray(r[field], dtype=float)


def collision_with_other(a, a_xyz, b, b_xyz, config):
    cc = config["collision"]
    arm = check_arm_envelope_xy_batch(
        xyz(a, "base_xyz_mm")[:2], a_xyz[:, :2],
        float(a["arm_envelope_radius_mm"]),
        xyz(b, "base_xyz_mm")[:2], b_xyz[:, :2],
        float(b["arm_envelope_radius_mm"]),
        float(cc["arm_clearance_mm"]),
        float(cc["geometry_epsilon_mm"]),
        bool(cc["touching_is_collision"]),
    )
    tcp_dist = np.linalg.norm(a_xyz[:, :2] - b_xyz[:, :2], axis=1)
    tcp_req = float(a["tcp_radius_mm"]) + float(b["tcp_radius_mm"])
    tcp_bad = tcp_dist <= tcp_req if cc["touching_is_collision"] else tcp_dist < tcp_req
    return ((np.asarray(arm.collision, dtype=bool) if cc["check_arm_envelope"] else False)
            | (tcp_bad if cc["check_tcp_radius"] else False))


def task_segments(robot, task, config):
    home = xyz(robot, "home_xyz_mm")
    start = np.asarray(task["start_xyz_mm"], dtype=float)
    end = np.asarray(task["end_xyz_mm"], dtype=float)
    speed_t = float(config["process"]["travel_speed_mm_s"])
    speed_d = float(config["process"]["deposition_speed_mm_s"])
    if min(speed_t, speed_d) <= 0:
        raise ValueError("Speeds must be positive")
    t = 0.0
    result = []
    for a, b, mode, speed in ((home, start, "T", speed_t),
                              (start, end, "D", speed_d),
                              (end, home, "T", speed_t)):
        dt = float(np.linalg.norm(b-a)) / speed
        if dt > 1e-10:
            result.append((t, t+dt, a.copy(), b.copy(), mode))
            t += dt
    if not result:
        raise ValueError("Zero-duration task")
    # XY reach: distance to base is convex along straight-line segments;
    # testing endpoints is sufficient for this radial bound.
    base = xyz(robot, "base_xyz_mm")
    reach = float(robot["xy_reach_radius_mm"])
    for point in (home, start, end):
        if np.linalg.norm(point[:2]-base[:2]) > reach + 1e-7:
            raise ValueError("XY reach exceeded")
    return result, t


def shifted(segments, delay):
    return [(t0+delay, t1+delay, a, b, mode)
            for t0, t1, a, b, mode in segments]


def trial_times(candidate, other, config):
    """Sample temporal grid plus segment endpoints, with displacement bound."""
    dt = float(config["simulation"]["max_time_step_s"])
    dx = float(config["simulation"]["max_tcp_step_mm"])
    if dt <= 0 or dx <= 0:
        raise ValueError("Simulation sampling settings must be positive")
    t0, t1 = candidate[0][0], candidate[-1][1]
    times = [t0, t1]
    for seg in candidate:
        a, b = seg[2], seg[3]
        n = max(1, math.ceil((seg[1]-seg[0])/dt),
                math.ceil(float(np.linalg.norm(b-a))/dx))
        times.extend(np.linspace(seg[0], seg[1], n+1))
    for seg in other:
        if t0 <= seg[0] <= t1:
            times.append(seg[0])
        if t0 <= seg[1] <= t1:
            times.append(seg[1])
    return np.unique(np.asarray(times, dtype=float))


def candidate_conflict(robot, candidate, robots, tracks, config):
    """Return (conflict, next plausible start). Includes idle HOME robots."""
    home = xyz(robot, "home_xyz_mm")
    finish = candidate[-1][1]
    for other in robots:
        if other["id"] == robot["id"]:
            continue
        rid = int(other["id"])
        existing = tracks[rid]
        times = trial_times(candidate, existing, config)
        a_xyz = positions_at_times(candidate, times, home)
        b_xyz = positions_at_times(existing, times, xyz(other, "home_xyz_mm"))
        bad = collision_with_other(robot, a_xyz, other, b_xyz, config)
        if np.any(bad):
            # If the other robot has an overlapping active job, retry after
            # its completion. If it is at HOME throughout the conflicting
            # interval, a delay cannot solve the geometry.
            hit = float(times[np.flatnonzero(bad)[0]])
            active_ends = [
                s[1] for s in existing
                if s[4] != "W" and s[0]-1e-8 <= hit <= s[1]+1e-8
            ]
            if active_ends:
                return True, max(active_ends) + float(config["simulation"]["max_time_step_s"])
            # A collision at a boundary might be with a job starting shortly;
            # skip its completion if it overlaps this candidate window.
            overlapping = [
                s[1] for s in existing
                if s[4] != "W" and s[0] <= finish and s[1] >= candidate[0][0]
            ]
            if overlapping:
                return True, max(overlapping) + float(config["simulation"]["max_time_step_s"])
            return True, None
    return False, None


def schedule(scenario, config, preferred, limit, max_retries):
    robots = sorted(config["robots"], key=lambda r: int(r["id"]))
    tracks = {int(r["id"]): [] for r in robots}
    available = {int(r["id"]): 0.0 for r in robots}
    tasks = scenario["tasks"][:limit] if limit else scenario["tasks"]
    changes, parallel = 0, 0
    for k, task in enumerate(tasks):
        choices = sorted(robots, key=lambda r: int(r["id"]) != int(preferred[k]))
        feasible = []
        reasons = []
        for robot in choices:
            rid = int(robot["id"])
            try:
                local, duration = task_segments(robot, task, config)
            except ValueError as exc:
                reasons.append(f"R{rid}: {exc}")
                continue
            start = available[rid]
            found = False
            for _ in range(max_retries):
                candidate = shifted(local, start)
                conflict, next_start = candidate_conflict(
                    robot, candidate, robots, tracks, config)
                if not conflict:
                    found = True
                    break
                if next_start is None or next_start <= start + 1e-9:
                    break
                start = next_start
            if found:
                feasible.append((start+duration, rid != int(preferred[k]),
                                 start, rid, candidate))
            else:
                reasons.append(f"R{rid}: blocked by collision/idle HOME or retry limit")
        if not feasible:
            raise RuntimeError(
                f"No safe candidate for task {k}: {'; '.join(reasons)}. "
                "No CSV exported."
            )
        # Prefer PPO robot when finish times tie; otherwise earlier completion.
        _, changed, start, rid, candidate = min(feasible, key=lambda x: (x[0], x[1]))
        if start > available[rid] + 1e-10:
            h = xyz(next(r for r in robots if r["id"] == rid), "home_xyz_mm")
            tracks[rid].append((available[rid], start, h.copy(), h.copy(), "W"))
        if any(start < available[oid]-1e-9 for oid in available if oid != rid):
            parallel += 1
        tracks[rid].extend(candidate)
        available[rid] = candidate[-1][1]
        changes += changed
        if (k+1) % 10 == 0 or k+1 == len(tasks):
            print(f"Scheduled {k+1}/{len(tasks)}; makespan={max(available.values()):.2f}s",
                  flush=True)
    makespan = max(available.values())
    for robot in robots:
        rid = int(robot["id"])
        if available[rid] < makespan - 1e-10:
            h = xyz(robot, "home_xyz_mm")
            tracks[rid].append((available[rid], makespan, h.copy(), h.copy(), "W"))
    return tracks, makespan, changes, parallel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tasks", type=int, default=30,
                    help="30 by default; 0 means all")
    ap.add_argument("--max-retries", type=int, default=30)
    ap.add_argument("--out", default="outputs/trajectory_PARALLEL_DIAGNOSTIC.csv")
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
    print("PPO internal makespan (all tasks):",
          round(float(np.max(env.robot_time)), 3), flush=True)
    limit = args.max_tasks if args.max_tasks > 0 else None
    tracks, makespan, changes, parallel = schedule(
        scenario, config, env.assignment, limit, args.max_retries)
    print("Tasks checked:", limit or len(scenario["tasks"]))
    print("PPO assignments changed:", changes)
    print("Tasks launched while another robot was active:", parallel)
    print("Parallel scheduler makespan:", round(makespan, 3), "s")
    print("Running complete sampled collision audit...", flush=True)
    count, first, margin = audit(tracks, config, makespan)
    print("Colliding pair-samples:", count)
    print("First collision:", first)
    print("Minimum arm safety margin:", round(margin, 3), "mm")
    if count:
        raise RuntimeError("Full audit failed. No CSV exported.")
    out = write_csv(tracks, args.out)
    print("Sampled collision-free diagnostic CSV:", out)
    print("NOT FINAL: contour-only; simplified travel; full Validator not passed.")


if __name__ == "__main__":
    main()
