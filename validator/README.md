# WAAM SB3 PPO Baseline

This is a deliberately simple first baseline for the 3-robot WAAM project.
PPO selects `[robot, deposition_task]`; the environment deterministically computes
travel time and deposition time and minimizes makespan.

## Install
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train
```bash
python train.py --scenario scenarios/example/scenario.json --timesteps 100000
```

## Evaluate
```bash
python evaluate.py --scenario scenarios/example/scenario.json --model checkpoints/ppo_waam_baseline
```

## Baseline design
- Observation: robot TCP/current time + all task endpoints/done flags + progress/makespan.
- Action: `MultiDiscrete([3, N])` = `(robot_index, task_index)`.
- Robot clocks are independent, so assignments represent parallel execution.
- Reward: negative increase in makespan, invalid repeated-task penalty, completion bonus.
- This baseline intentionally does NOT yet model continuous robot-robot collision,
  reach/workspace constraints, waits, safe parking, STL slicing, or Validator export.
  Those belong in the next environment version after the PPO scheduling baseline works.
