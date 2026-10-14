# WAAM Validator Report

## Overall Result

**PASS**

- WAAM Validator version: 1.0.1

## Input Summary

- Directory: `C:\Users\yenni\Downloads\v17_validation_20261010\02`
- Trajectory rows: 14308
- Target watertight: True

## Schedule

- Makespan: 1865.00 s
- R1: completion 1865.00 s; Deposition 259.28 s; Travel 143.82 s; Wait 1461.90 s
- R2: completion 1865.00 s; Deposition 256.14 s; Travel 144.12 s; Wait 1464.74 s
- R3: completion 1865.00 s; Deposition 1567.86 s; Travel 297.14 s; Wait 0.00 s

## Robot XY Reach

- Passed: True
- R1: max XY distance 1375.08 / 1500.00 mm; XY margin 124.92 mm; violating points 0
- R2: max XY distance 1334.77 / 1500.00 mm; XY margin 165.23 mm; violating points 0
- R3: max XY distance 1496.32 / 1500.00 mm; XY margin 3.68 mm; violating points 0

## Collision

- Arm Envelope events: 0
- Minimum Arm Envelope safety margin: 0.26 mm
- Arm centerline distance at worst case:
  250.26 mm
- Arm required centerline distance at worst case:
  250.00 mm
- TCP_RADIUS events: 0
- Minimum TCP distance: 250.26 mm

## Shape

- Target layer-integrated volume: 254007.549 mm³
- Deposited volume: 262218.103 mm³
- Intersection volume: 250568.870 mm³
- Underfill volume: 3438.674 mm³
- Overfill volume: 11649.233 mm³
- Coverage: 98.65%
- Underfill: 1.35%
- Overfill: 4.59%
- IoU: 94.32%
- Failed layers: 0 / 8

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
