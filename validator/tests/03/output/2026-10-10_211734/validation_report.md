# WAAM Validator Report

## Overall Result

**PASS**

- WAAM Validator version: 1.0.1

## Input Summary

- Directory: `C:\Users\yenni\Downloads\v17_validation_20261010\03`
- Trajectory rows: 7651
- Target watertight: True

## Schedule

- Makespan: 1259.23 s
- R1: completion 1259.23 s; Deposition 428.95 s; Travel 129.67 s; Wait 700.60 s
- R2: completion 1259.23 s; Deposition 282.63 s; Travel 149.58 s; Wait 827.01 s
- R3: completion 1259.23 s; Deposition 450.92 s; Travel 199.13 s; Wait 609.17 s

## Robot XY Reach

- Passed: True
- R1: max XY distance 1488.52 / 1500.00 mm; XY margin 11.48 mm; violating points 0
- R2: max XY distance 1491.03 / 1500.00 mm; XY margin 8.97 mm; violating points 0
- R3: max XY distance 1479.74 / 1500.00 mm; XY margin 20.26 mm; violating points 0

## Collision

- Arm Envelope events: 0
- Minimum Arm Envelope safety margin: 0.27 mm
- Arm centerline distance at worst case:
  250.27 mm
- Arm required centerline distance at worst case:
  250.00 mm
- TCP_RADIUS events: 0
- Minimum TCP distance: 250.89 mm

## Shape

- Target layer-integrated volume: 197274.975 mm³
- Deposited volume: 199073.453 mm³
- Intersection volume: 191627.145 mm³
- Underfill volume: 5647.808 mm³
- Overfill volume: 7446.299 mm³
- Coverage: 97.14%
- Underfill: 2.86%
- Overfill: 3.77%
- IoU: 93.60%
- Failed layers: 0 / 6

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
