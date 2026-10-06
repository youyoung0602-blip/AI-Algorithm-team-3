from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from algorithm.environment.waam_env import WAAMBaselineEnv
from algorithm.preprocessing.task_generator import build_scenario


def main():
    # =========================
    # 실행 인자
    # =========================
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stl",
        required=True,
        help="target.stl 경로"
    )

    p.add_argument(
        "--config",
        required=True,
        help="config.yaml 경로"
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=1000
    )

    p.add_argument(
        "--out",
        default="outputs/ppo_waam_baseline"
    )

    args = p.parse_args()

    # =========================
    # 1. STL + config → Scenario
    # =========================
    print("================================")
    print("Scenario 생성 시작")
    print("STL:", args.stl)
    print("CONFIG:", args.config)

    scenario = build_scenario(
        args.stl,
        args.config
    )

    print("Deposition Task 수:", len(scenario["tasks"]))
    print("Scenario 생성 완료")
    print("================================")

    # =========================
    # 2. WAAM Environment
    # =========================
    raw_env = WAAMBaselineEnv(scenario)

    print("Environment 생성 완료")

    # =========================
    # 3. SB3 환경 검사
    # =========================
    print("check_env 실행 중...")

    check_env(
        raw_env,
        warn=True
    )

    print("check_env 통과!")

    # =========================
    # 4. Monitor
    # =========================
    env = Monitor(raw_env)

    # =========================
    # 5. PPO 생성
    # =========================
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        device="auto",
        seed=42,
    )

    # =========================
    # 6. PPO 학습
    # =========================
    print("================================")
    print("PPO 학습 시작")
    print("================================")

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False
    )

    # =========================
    # 7. 모델 저장
    # =========================
    Path(args.out).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(args.out)

    print("================================")
    print("PPO 학습 완료")
    print("저장 위치:", args.out + ".zip")
    print("================================")


if __name__ == "__main__":
    main()