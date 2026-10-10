import argparse

from stable_baselines3 import PPO

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
        "--model",
        default="outputs/ppo_waam_baseline",
        help="학습된 PPO 모델 경로"
    )

    args = p.parse_args()

    # =========================
    # 1. Test scenario 생성
    # =========================
    print("================================")
    print("Evaluation 시작")
    print("STL:", args.stl)
    print("CONFIG:", args.config)
    print("MODEL:", args.model)
    print("================================")

    scenario = build_scenario(
        args.stl,
        args.config
    )

    print(
        "Deposition Task 수:",
        len(scenario["tasks"])
    )

    # =========================
    # 2. Environment 생성
    # =========================
    env = WAAMBaselineEnv(scenario)

    obs, _ = env.reset(seed=42)

    # =========================
    # 3. 학습된 PPO 불러오기
    # =========================
    model = PPO.load(args.model)

    print("PPO 모델 로드 완료")

    # =========================
    # 4. PPO 평가
    # =========================
    terminated = False
    truncated = False

    total_reward = 0.0
    step_count = 0

    while not (terminated or truncated):

        import torch

        # 현재 observation을 매 Step 새로 Tensor로 변환
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=model.device
        ).unsqueeze(0)

        # 현재 상태에서 PPO 확률 계산
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs

        # 500 Step마다 확률 확인
        if step_count % 500 == 0:
            print(
                f"Step {step_count} | "
                f"Action 확률: {probs.cpu().numpy()[0]} | "
                f"Robot 시간: {env.robot_time}"
            )

        # PPO 행동 선택
        action, _ = model.predict(
            obs,
            deterministic=False
        )

        # 환경 진행
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += reward
        step_count += 1

        if step_count % 500 == 0:
            print(
                f"완료 Task: {env.current_task}/{env.n_tasks} | "
                f"Makespan: {info['makespan']:.2f}s"
            )
    # =========================
    # 5. 최종 결과
    # =========================
    print("\n================================")
    print("Evaluation 완료")
    print("================================")

    print(
        "완료 Task:",
        env.current_task,
        "/",
        env.n_tasks
    )

    print(
        "전체 Step:",
        step_count
    )

    print(
        f"Total Reward: {total_reward:.3f}"
    )

    print(
        f"Final Makespan: {info['makespan']:.3f} sec"
    )

    print(
        "Robot별 완료시간:",
        env.robot_time.tolist()
    )

    print(
        "종료 상태:",
        "완료" if terminated else "중도 종료"
    )


if __name__ == "__main__":
    main()
