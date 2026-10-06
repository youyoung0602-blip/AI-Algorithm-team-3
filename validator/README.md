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
# Multi-Robot WAAM Scheduling with PPO

본 프로젝트는 다중 로봇 WAAM(Wire Arc Additive Manufacturing) 환경에서
**PPO(Proximal Policy Optimization)**와 **Adaptive Bundle Scheduling**을 결합하여
안전한 다중 로봇 적층 스케줄을 생성하는 것을 목표로 합니다.

최종 구조는 PPO가 로봇/작업 선택에 대한 선호도를 생성하고,
V15 Scheduler가 실제 Reach, Collision, 작업 시작 시간 등을 검사하여
최종 `trajectory.csv`를 생성하는 Hybrid 방식입니다.

---

# 1. PPO Algorithm

## 1.1 PPO의 역할

강화학습 알고리즘으로 Stable-Baselines3의 **PPO**를 사용합니다.

PPO가 최종 로봇 trajectory를 직접 생성하는 것이 아니라,
학습된 정책으로 task assignment를 생성하고 이를 V15 Scheduler의
robot preference로 사용합니다.

```text
STL
 ↓
Task Generator
 ↓
PPO
 ↓
Robot / Task Preference
 ↓
V15 Adaptive Bundle Scheduler
 ↓
Reachability Check
 ↓
Collision Check
 ↓
Earliest Safe Scheduling
 ↓
trajectory.csv
```

즉 최종 구조는 다음과 같습니다.

```text
PPO Preference
      +
Geometry-Aware Bundling
      +
Earliest Safe Finish
      +
Hard Safety Constraints
```

---

## 1.2 PPO Observation

PPO 환경에서는 로봇의 현재 위치와 작업시간,
각 task의 시작/종료 위치와 완료 여부,
전체 진행률 및 현재 makespan을 상태로 사용합니다.

핵심 코드는 다음과 같습니다.

```python
def _get_obs(self):
    obs = []

    # Robot state
    for i in range(3):
        obs.extend((self.robot_pos[i] / self.xyz_scale).tolist())
        obs.append(float(self.robot_time[i] / self.time_scale))

    # Task state
    for i in range(self.n_tasks):
        obs.extend((self.task_start[i] / self.xyz_scale).tolist())
        obs.extend((self.task_end[i] / self.xyz_scale).tolist())
        obs.append(float(self.task_done[i]))

    # Global state
    obs.append(float(self.task_done.mean()))
    obs.append(self._makespan() / self.time_scale)

    return np.asarray(obs, dtype=np.float32)
```

따라서 PPO는 단순히 로봇의 작업량만 보는 것이 아니라
현재 로봇 위치와 task 위치, 완료 상태 및 makespan을 함께 고려합니다.

---

# 2. Reward Function

PPO 학습의 핵심 목적은 **Makespan 증가를 최소화하는 것**입니다.

현재 Makespan은 다음과 같이 정의됩니다.

```python
def _makespan(self):
    return float(np.max(self.robot_time))
```

수식으로 표현하면

$$
M_t = \max(T_1,T_2,T_3)
$$

입니다.

각 action 수행 전후의 Makespan 차이를 이용하여 reward를 계산합니다.

```python
before = self._makespan()

travel_time = travel_dist / self.travel_speed
deposit_time = deposit_dist / self.deposition_speed

self.robot_time[robot_idx] += travel_time + deposit_time

after = self._makespan()

reward = -(after - before)
```

따라서 기본 Reward는

$$
r_t = -(M_{t+1}-M_t)
$$

입니다.

즉 현재 action으로 Makespan이 크게 증가할수록 더 큰 음의 보상을 받습니다.

### Completion Reward

모든 task가 완료되면 추가적인 완료 보상을 부여합니다.

```python
terminated = bool(np.all(self.task_done > 0.5))

if terminated:
    reward += self.finish_bonus
```

따라서 전체 reward 구조는 간단히

$$
r_t =
-\Delta M_t
+
R_{\text{finish}}
$$

로 표현할 수 있습니다.

이미 완료된 task를 다시 선택한 경우에는 `invalid_penalty`를 적용합니다.

```python
if self.task_done[task_idx] > 0.5:
    return (
        self._get_obs(),
        self.invalid_penalty,
        False,
        truncated,
        {"invalid_action": True}
    )
```

---

# 3. V15 Adaptive Bundle Scheduler

PPO의 결과를 그대로 trajectory로 사용하는 것이 아니라,
V15 Scheduler에서 실제 실행 가능한 경로로 변환합니다.

V15의 핵심 목표는

> **Collision과 Reach 조건을 만족하면서 최종 Makespan을 최소화하는 것**

입니다.

---

## 3.1 기존 방식의 문제

각 deposition chain마다 HOME으로 복귀하면 Travel 시간이 크게 증가합니다.

```text
HOME → Chain A → HOME
HOME → Chain B → HOME
HOME → Chain C → HOME
```

V15에서는 같은 layer의 인접 chain들을 하나의 Bundle로 구성합니다.

```text
HOME
 ↓
Chain A
 ↓
Direct Travel
 ↓
Chain B
 ↓
Direct Travel
 ↓
Chain C
 ↓
HOME
```

실제 V15에서는 두 chain 사이의 직접 이동이
HOME을 경유하는 것보다 짧은지 검사합니다.

```python
direct = np.linalg.norm(sb - ea)

via_home = (
    np.linalg.norm(ea - home)
    + np.linalg.norm(sb - home)
)

if direct < via_home:
    return True
```

이를 통해 불필요한 HOME 왕복 이동을 줄였습니다.

---

## 3.2 Earliest Safe Finish

각 Bundle에 대해 모든 실행 가능한 로봇을 검사합니다.

로봇이 현재 사용 가능한 시간부터 작업을 시작해 보고,
충돌이 발생하면 다음 안전한 시작시간을 탐색합니다.

```python
start = available[rid]

for _ in range(max_retries):

    candidate = shifted(local, start)

    conflict, next_start = candidate_conflict(
        robot,
        candidate,
        robots,
        tracks,
        config
    )

    if not conflict:
        found = True
        break

    refined = refine_earliest_collision_free_start(
        robot,
        local,
        robots,
        tracks,
        config,
        start,
        next_start,
        coarse_step=1.0,
        refine_iters=14,
    )

    start = refined if refined is not None else next_start
```

즉 단순히 "충돌하므로 실행 불가"로 끝내지 않고,

```text
현재 시작시간
     ↓
Collision?
     ↓ YES
다음 가능한 시간 탐색
     ↓
Boundary refinement
     ↓
Earliest Collision-Free Start
```

방식으로 가능한 가장 빠른 시작시간을 찾습니다.

---

## 3.3 Makespan 최소화를 위한 Robot 선택

안전하게 수행 가능한 후보는 다음과 같이 저장합니다.

```python
feasible.append((
    start + duration,   # finish time
    rid != pref,        # PPO preference
    start,
    rid,
    candidate,
    deposition_duration(candidate),
))
```

그리고 최종 선택은

```python
return min(
    feasible,
    key=lambda x: (x[0], x[1])
)
```

으로 이루어집니다.

즉 선택 우선순위는

```text
1순위 : 가장 빠른 Finish Time
2순위 : PPO Preferred Robot
```

입니다.

수식으로 표현하면

$$
r^*
=
\arg\min_{r\in R_{\mathrm{feasible}}}
F_r
$$

이며

$$
F_r = S_r + D_r
$$

입니다.

따라서 로봇별 작업량을 억지로 동일하게 만드는 것이 아니라,
**현재 조건에서 Bundle을 가장 빨리 끝낼 수 있는 안전한 로봇을 선택합니다.**

---

## 3.4 큰 작업부터 Scheduling

Bundle은 deposition 시간이 긴 순서대로 먼저 처리합니다.

```python
bundles = sorted(
    bundles,
    key=lambda b: bundle_dep_time(b, tasks, config),
    reverse=True
)
```

이는 긴 작업이 마지막에 남아 전체 Makespan을 증가시키는 현상을
줄이기 위한 scheduling 방식입니다.

---

## 3.5 Adaptive Bundle Split

Bundle 전체를 안전하게 실행할 수 없다면
해당 Bundle을 더 작은 단위로 분할합니다.

```python
choice = find_bundle_candidate(...)

if choice is None:

    pieces = split_bundle(bundle)

    left, right = pieces

    pending.insert(0, (right, depth + 1))
    pending.insert(0, (left, depth + 1))
```

또한 특정 로봇만 사용되는 경우에는
각 chain을 수행할 수 있는 로봇 집합이 변하는 위치를 기준으로
Bundle을 Adaptive하게 분할합니다.

```python
if sig != current_sig:
    out.append(current)
    current = [chain]
    current_sig = sig
```

따라서 Test 번호별로 별도의 규칙을 넣지 않고,
실제 geometry와 robot reachability를 기준으로 작업을 분할합니다.

---

# 4. V15 Makespan Minimization Strategy

최종적으로 V15에서는 다음 5가지 방법을 조합하여 Makespan을 감소시킵니다.

| Strategy | Purpose |
|---|---|
| Direct Travel Bundle | HOME 왕복 Travel 감소 |
| Longest Bundle First | 긴 작업이 마지막에 남는 현상 감소 |
| Earliest Safe Start | 불필요한 Wait 감소 |
| Earliest Finish Robot | 가장 빠르게 완료 가능한 Robot 선택 |
| Adaptive Bundle Split | 여러 Robot의 병렬 작업 증가 |

전체적인 흐름은 다음과 같습니다.

```text
Deposition Chains
       ↓
Same-Layer Bundling
       ↓
Direct Travel 적용
       ↓
Long Bundle 우선 Scheduling
       ↓
각 Robot의 Earliest Safe Start 계산
       ↓
Earliest Finish Robot 선택
       ↓
실패 시 Bundle Split
       ↓
Parallel Scheduling
       ↓
Final Collision Audit
       ↓
trajectory.csv
```

---

# 5. Results

최종 **V5 Task Generator + V15 Adaptive Bundle Scheduler**를
동일하게 Test 01~06에 적용했습니다.

| Test | Makespan | Shape IoU | Collision | Validator |
|---|---:|---:|---:|---|
| 01 | - | - | 0 | **PASS** |
| 02 | 1,865.00 s | 94.32% | 0 | **PASS** |
| 03 | 1,263.15 s | 93.60% | 0 | **PASS** |
| 04 | 37,918.78 s | 94.09% | 0 | **PASS** |
| 05 | 100,522.34 s | 90.38% | 0 | **PASS** |
| 06 | 5,209.81 s | 95.25% | 0 | **PASS** |

**Test 01~06 모두 공식 WAAM Validator를 통과했습니다.**

---

## Test 04 Makespan Improvement

Test 04에서 기존 V12 Scheduler의 Makespan은

$$
49,758.38s
$$

였으며 V15에서는

$$
37,918.78s
$$

로 감소했습니다.

따라서

$$
\frac{49758.38-37918.78}{49758.38}\times100
=
23.79\%
$$

즉 **약 23.79%의 Makespan 감소**를 달성했습니다.

---

## Test 05 Large-Scale Result

Test 05에서도 공식 Validator를 통과했습니다.

| Metric | Result |
|---|---:|
| Makespan | 100,522.34 s |
| Coverage | 91.55% |
| IoU | 90.38% |
| Failed Layers | 0 / 350 |
| ARM Collision | 0 |
| TCP Collision | 0 |
| Reach Violation | 0 |
| Validator | **PASS** |

---

# 6. Conclusion

본 프로젝트에서는 PPO가 학습 기반의 Robot/Task Preference를 제공하고,
V15 Adaptive Bundle Scheduler가 실제 물리적 제약을 고려하여
최종 trajectory를 생성하도록 구성했습니다.

특히 V15에서는

**Geometry-Aware Bundling + Direct Travel + Earliest Safe Start +
Earliest Finish Robot Selection + Adaptive Split**

을 적용하여 불필요한 Travel과 Wait를 감소시키고
Multi-Robot Parallelism을 증가시켰습니다.

그 결과 서로 다른 크기와 형상을 가진 Test 01~06에서
동일한 알고리즘으로 공식 Validator PASS를 달성했습니다.
