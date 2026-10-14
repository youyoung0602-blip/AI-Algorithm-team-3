# PPO 기반 다중 로봇 WAAM 경로 생성 및 안전 스케줄링

Task Generator로 STL 형상을 적층 작업으로 변환하고, PPO로 로봇 배정 선호도를 생성한 뒤 Scheduler v17로 안전한 작업 일정을 구성합니다.

## 모델별 적용 결과

시간 형식은 `DD:HH:MM:SS`(일:시:분:초)입니다. Makespan은 초 단위로 반올림했습니다.

| 모델 | Makespan (초) | Makespan (DD:HH:MM:SS) | 계산 시간 (DD:HH:MM:SS) | Validator 결과 |
| --- | ---: | --- | --- | --- |
| 01 · Honeycomb | 683.98 | 00:00:11:24 | 00:00:46:00 | PASS |
| 02 · Spiral | 1,865.00 | 00:00:31:05 | 00:02:38:00 | PASS |
| 03 · Triangular truss | 1,259.23 | 00:00:20:59 | 00:01:08:00 | PASS |
| 04 · Motor inner | 37,094.55 | 00:10:18:15 | 00:05:31:00 | PASS |
| 05 · Twisted shell | 100,522.34 | 01:03:55:22 | 00:17:41:00 | PASS |
| 06 · Angle frame | 5,189.37 | 00:01:26:29 | 00:02:35:00 | PASS |

Makespan은 로봇 일정의 예정 작업 시간입니다. 계산 시간은 trajectory 생성에 걸린 대략적인 경과 시간으로, 공식 Validator 실행 시간은 제외합니다.

## Validator 적용 결과

### 간트차트

Test 1부터 Test 6까지의 결과입니다.

#### Test 01

![Test 01 간트차트](assets/gantt_01.png)

#### Test 02

![Test 02 간트차트](assets/gantt_02.png)

#### Test 03

![Test 03 간트차트](assets/gantt_03.png)

#### Test 04

![Test 04 간트차트](assets/gantt_04.png)

#### Test 05

![Test 05 간트차트](assets/gantt_05.png)

#### Test 06

![Test 06 간트차트](assets/gantt_06.png)

### 검증 결과 화면

#### Test 01

![Test 01 검증 결과](assets/validation_01.png)

#### Test 02

![Test 02 검증 결과](assets/validation_02.png)

#### Test 03

![Test 03 검증 결과](assets/validation_03.png)

#### Test 04

![Test 04 검증 결과](assets/validation_04.png)

#### Test 05

![Test 05 검증 결과](assets/validation_05.png)

#### Test 06

![Test 06 검증 결과](assets/validation_06.png)

## 알고리즘 수도코드

```text
Algorithm: Task Generator + PPO-Clip + Scheduler v17
01 Input: STL M, 설정 C, 정책 πθ; tasks T←∅

A. Task Generator
02 For each layer ℓ: Aℓ←Slice(M,zℓ), zℓ=zbase+(ℓ+η)h
03     ℘ℓ←Hatch 후보 ∪ Skeleton 후보
04     For each P∈℘ℓ: D←Buffer(P,w/2); L←Σe∈P‖eend−estart‖₂
05         cov←|Aℓ∩D|/|Aℓ|; over←|D\Aℓ|/|Aℓ|; iou←|Aℓ∩D|/|Aℓ∪D|
06     F←{P: cov≥cmin ∧ over≤omax ∧ iou≥imin}; F=∅이면 종료
07     U←{P∈F: iou≥i*}; U=∅이면 U←F
08     P*←argminP∈U [L+0.05w|P|+5max(0,i*−iou)max(L,w)]
09     P*의 선분에 높이 zℓ를 부여하여 순서대로 T에 추가

B. PPO-Clip Training — 재학습 시에만 수행
10 정책 πθ·가치함수 Vφ 초기화; 기존 모델 사용 시 20행으로 이동
11 For each update: πold←πθ; rollout 수집, old log-probability·가치 저장
12     δt←rt+1+γ(1−dt+1)Vold(st+1)−Vold(st)
13     Ât←δt+γλ(1−dt+1)Ât+1 (역순); Vt_tar←Ât+Vold(st)
14     For K epochs and each shuffled mini-batch:
15         ρt←exp(log πθ(at|st)−log πold(at|st))
16         Lclip←mean[min(ρtÂt,clip(ρt,1−ε,1+ε)Ât)]
17         LV←mean[(Vt_tar−Vφ(st))²]; H←mean[−Σaπθ(a|st)log πθ(a|st)]
18         Ltotal←−Lclip+cvLV−ceH
19         (θ,φ)←(θ,φ)−αlr∇θ,φLtotal; old log-probability·Ât·Vt_tar는 고정

C. PPO Inference
20 학습된 정책을 준비하고 환경 초기화; For each task t∈T:
21     Fr,t←τr+‖xr−pt,start‖₂/vT+‖pt,end,xy−pt,start,xy‖₂/vD
22     at←argmaxaπθ(a|st); preft←argsortr(Fr,t)[at]; 환경 상태 갱신

D. Scheduler v17 + Validator
23 Sbest←충돌 검사를 통과한 V15 기본 일정; Cmax(S)←maxrCr(S)
24 Q←선분을 chain으로 연결; 공통 도달 가능 로봇이 없어지기 직전에 분할
25 V←동일 층 bundle 구성 및 생성 가능한 adaptive bundle 변형 목록
26 For k=0,...,19: B←V[k mod |V|]; 비었거나 |B|>300이면 continue
27     Ok←적층시간 오름차순(k=0), 내림차순(k=1), 순서 교환(k≥2)
28     S←최소 완료 안전 배정(Ok(B),pref); 충돌 시 지연, 배치 불가 시 분할
29     S가 존재하고 Cmax(S)<Cmax(Sbest)이며 충돌 검사 통과 시 Sbest←S
30 최종 충돌 검사 → D/T/W CSV 저장 → 공식 Validator로 결과 평가
```

$A_\ell$: 목표 단면, $D$: 비드 영역, $w$: 비드 폭, $L$: 경로 길이, $|P|$: 선분 수입니다.

PPO의 action은 로봇 ID가 아닌 **예상 완료 시각 순위**이며, Scheduler에 로봇 선호도로 전달됩니다. 이번 검증은 저장된 PPO 모델을 이용한 추론으로 수행했으며, 수도코드의 PPO-Clip 학습 단계는 재학습 시에만 수행합니다.

## 이 문제에 적용한 이유

### Task Generator

STL에는 물체 표면의 형상 정보가 있지만 로봇이 따라갈 적층 선분은 포함되어 있지 않습니다. 따라서 층별 단면을 추출하고 비드 폭을 고려하여 실제 적층 경로를 생성해야 합니다.

Task Generator는 단면마다 형상과 비용을 평가하여 경로를 선택합니다. 3차원 형상을 한 번에 비교하는 대신 각 층에서 경로 후보를 평가할 수 있다는 점을 활용했습니다.

### PPO

같은 선분이라도 로봇의 현재 위치와 누적 작업 시간에 따라 적합한 로봇이 달라집니다. 당장 유리한 배정이 이후 작업에서도 유리하다고 보장할 수는 없습니다.

현재 상태와 이후 누적 보상을 고려하는 PPO를 활용하여 상태에 따른 로봇 배정 선호도를 제공하고자 했습니다.

## 알고리즘의 특징과 장단점

| 구성 요소 | 특징 | 장점 | 한계 |
| --- | --- | --- | --- |
| Task Generator | 층별 단면에서 후보를 생성하고, 형상 제약을 만족하는 경로 중 비용이 작은 경로 선택 | Coverage·Overfill·IoU를 고려하면서 경로 길이와 선분 분절을 줄일 수 있음 | 후보를 반복 평가하므로 계산량이 큼 |
| PPO | 로봇 위치·누적 작업 상태를 바탕으로 예상 완료 시각 순위 중 하나를 선택 | 고정 규칙 대신 상태에 따른 배정 선호도를 학습할 수 있음 | 학습 조건과 다른 형상에서도 성능이 유지되는지 확인 필요 |
| Scheduler v17 · Bundle | 도달 범위에 맞게 chain을 분할하고 bundle 처리 순서를 탐색 | 선분별 배정 부담을 줄이고, 연속 작업을 통해 HOME 왕복과 이동을 줄일 수 있음 | 제한된 순서 탐색이므로 전역 최적해를 보장하지 않음 |
