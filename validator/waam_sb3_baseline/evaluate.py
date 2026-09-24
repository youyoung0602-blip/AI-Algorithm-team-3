import argparse
from stable_baselines3 import PPO
from algorithm.environment.waam_env import WAAMBaselineEnv, load_scenario


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="scenarios/example/scenario.json")
    p.add_argument("--model", default="checkpoints/ppo_waam_baseline")
    args = p.parse_args()

    env = WAAMBaselineEnv(load_scenario(args.scenario))
    obs, _ = env.reset(seed=42)
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action, _ = PPO.load(args.model).predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    env.render()
    print(f"total_reward={total_reward:.3f}")
    print(f"final_makespan={info['makespan']:.3f}s")


if __name__ == "__main__":
    main()
