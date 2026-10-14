# WAAM Validator Report

## Overall Result

**PASS**

- WAAM Validator version: 1.0.1

## Input Summary

- Directory: `C:\Users\yenni\Downloads\v17_validation_20261010\01`
- Trajectory rows: 4861
- Target watertight: True

## Schedule

- Makespan: 683.98 s
- R1: completion 683.98 s; Deposition 132.36 s; Travel 104.55 s; Wait 447.07 s
- R2: completion 683.98 s; Deposition 8.92 s; Travel 22.68 s; Wait 652.38 s
- R3: completion 683.98 s; Deposition 456.49 s; Travel 152.06 s; Wait 75.43 s

## Robot XY Reach

- Passed: True
- R1: max XY distance 1472.45 / 1500.00 mm; XY margin 27.55 mm; violating points 0
- R2: max XY distance 1475.49 / 1500.00 mm; XY margin 24.51 mm; violating points 0
- R3: max XY distance 1475.35 / 1500.00 mm; XY margin 24.65 mm; violating points 0

## Collision

- Arm Envelope events: 0
- Minimum Arm Envelope safety margin: 3.70 mm
- Arm centerline distance at worst case:
  253.70 mm
- Arm required centerline distance at worst case:
  250.00 mm
- TCP_RADIUS events: 0
- Minimum TCP distance: 253.70 mm

## Shape

- Target layer-integrated volume: 116840.445 mm³
- Deposited volume: 116983.723 mm³
- Intersection volume: 112142.396 mm³
- Underfill volume: 4698.033 mm³
- Overfill volume: 4841.325 mm³
- Coverage: 95.98%
- Underfill: 4.02%
- Overfill: 4.14%
- IoU: 92.16%
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
