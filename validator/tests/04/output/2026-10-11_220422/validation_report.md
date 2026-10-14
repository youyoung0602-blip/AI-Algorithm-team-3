# WAAM Validator Report

## Overall Result

**PASS**

- WAAM Validator version: 1.0.1

## Input Summary

- Directory: `C:\Users\yenni\Downloads\v17_validation_20261010\04`
- Trajectory rows: 9237
- Target watertight: True

## Schedule

- Makespan: 37094.55 s
- R1: completion 37094.55 s; Deposition 16511.33 s; Travel 1236.59 s; Wait 19346.63 s
- R2: completion 37094.55 s; Deposition 19239.75 s; Travel 1481.86 s; Wait 16372.95 s
- R3: completion 37094.55 s; Deposition 9890.86 s; Travel 860.21 s; Wait 26343.48 s

## Robot XY Reach

- Passed: True
- R1: max XY distance 1469.96 / 1500.00 mm; XY margin 30.04 mm; violating points 0
- R2: max XY distance 1486.28 / 1500.00 mm; XY margin 13.72 mm; violating points 0
- R3: max XY distance 1494.74 / 1500.00 mm; XY margin 5.26 mm; violating points 0

## Collision

- Arm Envelope events: 0
- Minimum Arm Envelope safety margin: 0.15 mm
- Arm centerline distance at worst case:
  250.15 mm
- Arm required centerline distance at worst case:
  250.00 mm
- TCP_RADIUS events: 0
- Minimum TCP distance: 250.15 mm

## Shape

- Target layer-integrated volume: 6661016.843 mm³
- Deposited volume: 6286561.712 mm³
- Intersection volume: 6276570.748 mm³
- Underfill volume: 384446.042 mm³
- Overfill volume: 9990.964 mm³
- Coverage: 94.23%
- Underfill: 5.77%
- Overfill: 0.15%
- IoU: 94.09%
- Failed layers: 0 / 50

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
