import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from algorithm.environment.waam_env import WAAMBaselineEnv, load_scenario


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenarios/example/scenario.json")
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--out", default="checkpoints/ppo_waam_baseline")
    args = p.parse_args()

    scenario = load_scenario(args.scenario)
    raw_env = WAAMBaselineEnv(scenario)
    check_env(raw_env, warn=True)
    env = Monitor(raw_env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="runs/",
        device="auto",
        seed=42,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved model: {args.out}.zip")


if __name__ == "__main__":
    main()
