from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET

source = Path(r'C:\Users\yenni\OneDrive\문서\[INE4107]중간보고서(양식).docx')
out = Path(r'C:\Users\yenni\Downloads\WAAM_GitHub_README')
assets = out / 'assets'
assets.mkdir(parents=True, exist_ok=True)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(source) as doc:
    root = ET.fromstring(doc.read('word/document.xml'))
    paragraphs = [''.join(t.text or '' for t in p.findall('.//w:t', ns)) for p in root.findall('.//w:p', ns)]
    for i in range(1, 13):
        name = f'gantt_{i:02d}.png' if i <= 6 else f'validation_{i-6:02d}.png'
        (assets / name).write_bytes(doc.read(f'word/media/image{i}.png'))

start = paragraphs.index('Algorithm: Task Generator + PPO-Clip + Scheduler v17')
end = next(i for i in range(start, len(paragraphs)) if paragraphs[i].startswith('30 최종 충돌 검사'))
pseudocode = '\n'.join(paragraphs[start:end+1])
text = '''# PPO 기반 다중 로봇 WAAM 경로 생성 및 안전 스케줄링

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

'''
for i in range(1, 7):
    text += f'#### Test {i:02d}\n\n![Test {i:02d} 간트차트](assets/gantt_{i:02d}.png)\n\n'
text += '### 검증 결과 화면\n\n'
for i in range(1, 7):
    text += f'#### Test {i:02d}\n\n![Test {i:02d} 검증 결과](assets/validation_{i:02d}.png)\n\n'
text += '## 알고리즘 수도코드\n\n```text\n' + pseudocode + '\n```\n\n'
text += r'''$A_\ell$: 목표 단면, $D$: 비드 영역, $w$: 비드 폭, $L$: 경로 길이, $|P|$: 선분 수입니다.

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
'''
(out / 'README.md').write_text(text, encoding='utf-8')
assert all(x not in text for x in ['고유영', '2021040673', 'youyoung0602', 'https://'])
assert len(list(assets.glob('*.png'))) == 12
with ZipFile(out.with_suffix('.zip'), 'w', ZIP_DEFLATED) as archive:
    for item in sorted(out.rglob('*')):
        if item.is_file():
            archive.write(item, item.relative_to(out))
print(out / 'README.md')
print(out.with_suffix('.zip'))
print('Verified: 12 images, original pseudocode, no personal header or external URLs')
