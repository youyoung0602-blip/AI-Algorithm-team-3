import os
import subprocess
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

repo = Path(r'C:\Users\yenni\OneDrive\바탕 화면\AI-Algorithm-team-3')
baseline = repo / 'validator' / 'waam_sb3_baseline'
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / ('v17_validation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir(exist_ok=True)
env = os.environ.copy()
env['PYTHONPATH'] = os.pathsep.join(str(p) for p in [repo / 'validator' / 'src', baseline, baseline / 'schedulers', baseline / 'generators', baseline / 'training'])
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['PYTHONIOENCODING'] = 'utf-8'
selected = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else {'01', '02', '03', '06'}
result_file = out / ('results_' + '_'.join(sorted(selected)) + '.json')
results = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else []
print(f'OUTPUT: {out}', flush=True)
for case in sorted((repo / 'test_data').iterdir()):
    if case.name not in selected or not case.is_dir() or not (case / 'target.stl').exists():
        continue
    if any(r['case'] == case.name for r in results):
        continue
    job = out / case.name
    job.mkdir(exist_ok=True)
    for name in ['config.yaml', 'target.stl']:
        shutil.copy2(case / name, job / name)
    cmd = [sys.executable, '-B', str(Path(__file__).parent / 'v17_experiment' / 'safe_scheduler_v17_reach_split.py'), '--stl', str(job / 'target.stl'), '--config', str(job / 'config.yaml'), '--model', str(baseline / 'outputs' / 'ppo_fixed_v1.zip'), '--out', str(job / 'trajectory.csv')]
    print(f'CASE {case.name}: generating V17, 20 iterations, all tasks', flush=True)
    with (job / 'scheduler.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, env=env, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = proc.wait()
    row = {'case': case.name, 'generation_exit': code}
    if code == 0:
        print(f'CASE {case.name}: official validator', flush=True)
        p = subprocess.run([sys.executable, '-B', '-m', 'waam_validator.cli', 'run', str(job)], env=env, cwd=out, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (job / 'validator.log').write_text(p.stdout + p.stderr, encoding='utf-8')
        print(p.stdout + p.stderr, flush=True)
        row['validator_exit'] = p.returncode
        summaries = list((job / 'output').glob('*/summary.json'))
        if summaries:
            row['summary'] = json.loads(summaries[-1].read_text(encoding='utf-8'))
    results.append(row)
    result_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print('COMPLETE', flush=True)
