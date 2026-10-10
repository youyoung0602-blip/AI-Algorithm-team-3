import argparse
from pathlib import Path
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from task_generator_v2 import build_scenario
from algorithm.environment.collision_aware_waam_env import CollisionAwareWAAMEnv


def enriched(stl, config_path):
    scenario=build_scenario(stl,config_path)
    with open(config_path,encoding="utf-8") as f: cfg=yaml.safe_load(f)
    by_id={int(r["id"]):r for r in cfg["robots"]}
    for r in scenario["robots"]:
        src=by_id[int(r["robot_id"])]
        for key in ("xy_reach_radius_mm","arm_envelope_radius_mm","tcp_radius_mm"):
            r[key]=float(src[key])
    scenario["collision"]=dict(cfg["collision"])
    scenario["collision_reward"]={"makespan":1.0,"travel":0.30,"imbalance":0.50,"collision":1000.0,"near":10.0,"near_margin_mm":50.0,"finish_bonus":20.0}
    return scenario


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--case",action="append",nargs=2,metavar=("STL","CONFIG"),required=True,help="Repeat for multiple geometries")
    ap.add_argument("--steps-per-case",type=int,default=100000)
    ap.add_argument("--cycles",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",default="outputs/ppo_collision_v1")
    args=ap.parse_args()
    model=None
    for cycle in range(args.cycles):
        for idx,(stl,cfg) in enumerate(args.case,1):
            print(f"Training cycle {cycle+1}/{args.cycles}, case {idx}/{len(args.case)}: {stl}",flush=True)
            env=Monitor(CollisionAwareWAAMEnv(enriched(stl,cfg)))
            if model is None:
                model=PPO("MlpPolicy",env,verbose=1,seed=args.seed,n_steps=2048,batch_size=64,gamma=0.99,learning_rate=3e-4)
            else:
                model.set_env(env)
            model.learn(total_timesteps=args.steps_per_case,reset_num_timesteps=False,progress_bar=True)
            env.close()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    model.save(args.out)
    print("Saved:",args.out+".zip",flush=True)

if __name__=="__main__": main()
