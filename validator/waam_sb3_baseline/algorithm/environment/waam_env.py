from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def load_scenario(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class WAAMBaselineEnv(gym.Env):
    """Simple SB3/Gymnasium baseline for 3-robot WAAM scheduling.

    PPO decides only (robot, deposition_task). Geometry/path generation is kept
    deterministic so the first baseline learns assignment + ordering rather
    than continuous TCP control.

    Action
    ------
    MultiDiscrete([3, N]): [robot_index, task_index]

    Observation
    -----------
    Flattened normalized vector:
      robots: 3 * [x, y, z, available_time]
      tasks : N * [sx, sy, sz, ex, ey, ez, done]
      global: [progress, makespan]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: dict[str, Any]):
        super().__init__()
        self.scenario = scenario
        self.robots_cfg = sorted(scenario["robots"], key=lambda r: int(r["robot_id"]))
        if [int(r["robot_id"]) for r in self.robots_cfg] != [1, 2, 3]:
            raise ValueError("Baseline assumes validator robot IDs 1, 2, 3.")
        self.tasks_cfg = scenario["tasks"]
        self.n_tasks = len(self.tasks_cfg)
        if self.n_tasks < 1:
            raise ValueError("At least one deposition task is required.")

        process = scenario["process"]
        self.travel_speed = float(process["travel_speed_mm_s"])
        self.deposition_speed = float(process["deposition_speed_mm_s"])
        if self.travel_speed <= 0 or self.deposition_speed <= 0:
            raise ValueError("Speeds must be positive.")

        self.xyz_scale = float(scenario.get("normalization_radius_mm", 2000.0))
        self.time_scale = float(scenario.get("time_normalization_s", 1000.0))
        reward_cfg = scenario.get("reward", {})
        self.invalid_penalty = float(reward_cfg.get("invalid", -10.0))
        self.finish_bonus = float(reward_cfg.get("finish_bonus", 10.0))
        self.max_steps = int(scenario.get("max_steps", max(20, 4 * self.n_tasks)))

        # Vanilla SB3 PPO supports MultiDiscrete directly.
        self.action_space = spaces.MultiDiscrete([3, self.n_tasks])
        obs_dim = 3 * 4 + self.n_tasks * 7 + 2
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        self.robot_pos: np.ndarray
        self.robot_time: np.ndarray
        self.task_start: np.ndarray
        self.task_end: np.ndarray
        self.task_done: np.ndarray
        self.assignment: np.ndarray
        self.step_count = 0

    def _initial_arrays(self) -> None:
        self.robot_pos = np.asarray([
            r.get("home_xyz_mm", r["base_xyz_mm"]) for r in self.robots_cfg
        ], dtype=np.float32)
        self.robot_time = np.zeros(3, dtype=np.float32)
        self.task_start = np.asarray([t["start_xyz_mm"] for t in self.tasks_cfg], dtype=np.float32)
        self.task_end = np.asarray([t["end_xyz_mm"] for t in self.tasks_cfg], dtype=np.float32)
        self.task_done = np.zeros(self.n_tasks, dtype=np.float32)
        self.assignment = np.full(self.n_tasks, -1, dtype=np.int32)

    def _makespan(self) -> float:
        return float(np.max(self.robot_time))

    def _get_obs(self) -> np.ndarray:
        obs: list[float] = []
        for i in range(3):
            obs.extend((self.robot_pos[i] / self.xyz_scale).tolist())
            obs.append(float(self.robot_time[i] / self.time_scale))
        for i in range(self.n_tasks):
            obs.extend((self.task_start[i] / self.xyz_scale).tolist())
            obs.extend((self.task_end[i] / self.xyz_scale).tolist())
            obs.append(float(self.task_done[i]))
        obs.append(float(self.task_done.mean()))
        obs.append(self._makespan() / self.time_scale)
        return np.asarray(obs, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._initial_arrays()
        self.step_count = 0
        return self._get_obs(), {"makespan": 0.0}

    def step(self, action):
        robot_idx, task_idx = int(action[0]), int(action[1])
        self.step_count += 1

        # Vanilla PPO has no action mask: repeated tasks receive a penalty.
        if self.task_done[task_idx] > 0.5:
            truncated = self.step_count >= self.max_steps
            return self._get_obs(), self.invalid_penalty, False, truncated, {
                "invalid_action": True,
                "makespan": self._makespan(),
            }

        before = self._makespan()
        travel_dist = float(np.linalg.norm(self.task_start[task_idx] - self.robot_pos[robot_idx]))
        deposit_dist = float(np.linalg.norm(
            self.task_end[task_idx, :2] - self.task_start[task_idx, :2]
        ))
        travel_time = travel_dist / self.travel_speed
        deposit_time = deposit_dist / self.deposition_speed

        # Each robot has its own clock, allowing parallel schedules.
        self.robot_time[robot_idx] += travel_time + deposit_time
        self.robot_pos[robot_idx] = self.task_end[task_idx]
        self.task_done[task_idx] = 1.0
        self.assignment[task_idx] = robot_idx + 1

        after = self._makespan()
        reward = -(after - before)
        terminated = bool(np.all(self.task_done > 0.5))
        if terminated:
            reward += self.finish_bonus
        truncated = self.step_count >= self.max_steps and not terminated

        info = {
            "invalid_action": False,
            "robot_id": robot_idx + 1,
            "task_id": int(self.tasks_cfg[task_idx].get("task_id", task_idx)),
            "travel_time": travel_time,
            "deposition_time": deposit_time,
            "makespan": after,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        print(f"makespan={self._makespan():.3f}s, done={int(self.task_done.sum())}/{self.n_tasks}")
        print("robot times:", self.robot_time.tolist())
        print("assignment:", self.assignment.tolist())
