# WAAM Validator Report

## Overall Result

**PASS**

- WAAM Validator version: 1.0.1

## Input Summary

- Directory: `C:\Users\yenni\Downloads\v17_validation_20261010\05`
- Trajectory rows: 131992
- Target watertight: True

## Schedule

- Makespan: 100522.34 s
- R1: completion 100522.34 s; Deposition 82013.69 s; Travel 10789.27 s; Wait 7719.38 s
- R2: completion 100522.34 s; Deposition 69700.49 s; Travel 8710.15 s; Wait 22111.71 s
- R3: completion 100522.34 s; Deposition 62504.60 s; Travel 6966.05 s; Wait 31051.69 s

## Robot XY Reach

- Passed: True
- R1: max XY distance 1499.20 / 1500.00 mm; XY margin 0.80 mm; violating points 0
- R2: max XY distance 1490.79 / 1500.00 mm; XY margin 9.21 mm; violating points 0
- R3: max XY distance 1499.61 / 1500.00 mm; XY margin 0.39 mm; violating points 0

## Collision

- Arm Envelope events: 0
- Minimum Arm Envelope safety margin: 0.05 mm
- Arm centerline distance at worst case:
  250.05 mm
- Arm required centerline distance at worst case:
  250.00 mm
- TCP_RADIUS events: 0
- Minimum TCP distance: 250.05 mm

## Shape

- Target layer-integrated volume: 31086145.215 mm³
- Deposited volume: 28862515.052 mm³
- Intersection volume: 28459917.397 mm³
- Underfill volume: 2626227.785 mm³
- Overfill volume: 402597.654 mm³
- Coverage: 91.55%
- Underfill: 8.45%
- Overfill: 1.30%
- IoU: 90.38%
- Failed layers: 0 / 350

## Failure Reasons

None

## Warnings

None

## Validation Violations

These findings contribute to a normal `FAIL`; they do not mean that the pipeline ended with
the fatal status `ERROR`.

None

## Output Files

- `summary.json`
- `validation_report.md`
- `robot_metrics.csv`
- `collision_events.csv`
- `layer_metrics.csv`
- `run.log`
- `validation_inputs.json`

## Interpretation Limitations

본 검증기는 각 로봇을 폭이 고정된 Base–TCP 2D Capsule로 단순화하고, XY 평면상 Capsule 안전 여유와 TCP 허용 원 침범만을 로봇 간 충돌로 판단한다. 검사는 adaptive sample 기반이며 연속시간 swept collision을 보증하지 않는다. 실제 로봇 링크, 관절 자세, Z 방향 분리, 지그 및 환경 충돌은 반영하지 않는다. 형상 검증은 일정한 비드 폭과 layer 높이를 가정한 명목 기하 모델이며 열변형, 비드 형상 변화, 용융풀 거동 및 공정 불안정성을 예측하지 않는다.
