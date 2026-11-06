"""Freeze/verify prior research files without rewriting prior experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
REPORT = ROOT / 'research/round4070_comprehensive_traversability'
SNAPSHOT = OUT / 'prior_work_snapshot.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def paths():
    result = {ROOT / 'lift_curtain/perception/online_occupancy.py'}
    for folder in ['scripts/round4070_f', 'research/round4070_learning_traversability',
                   'outputs/round4070_learning_traversability']:
        result.update(p for p in (ROOT / folder).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts)
    e = load(ROOT / 'outputs/round4070_learning_traversability/dataset/manifest.json')
    for episode in e['episodes']:
        for filename in episode['source_sha256']:
            result.add(ROOT / 'outputs/round4070_fixed_probe_benchmark/trials' /
                       episode['trial_id'] / filename)
    # Record the earlier written research evidence and runner sources too.
    for parent in ['research', 'scripts']:
        for folder in (ROOT / parent).glob('round4070*'):
            if not folder.is_dir() or folder.name in ['round4070_g', 'round4070_comprehensive_traversability']:
                continue
            result.update(p for p in folder.rglob('*') if p.is_file()
                          and p.suffix in ['.md', '.py', '.ps1', '.json', '.csv']
                          and '__pycache__' not in p.parts)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        if SNAPSHOT.exists():
            raise SystemExit('Snapshot already exists; use verification, never overwrite the reference.')
        files = paths()
        state = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                     start_head=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                     remote=git('remote', 'get-url', 'origin'),
                     original_status=git('status', '--porcelain=v1'),
                     files={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files},
                     protected_bytes=sum(p.stat().st_size for p in files))
        SNAPSHOT.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    state = load(SNAPSHOT)
    changes = [name for name, digest in state['files'].items()
               if not (ROOT / name).exists() or sha(ROOT / name) != digest]
    result = dict(status='PASS' if not changes else 'FAIL',
                  checked_utc=datetime.now(timezone.utc).isoformat(),
                  baseline_head=state['start_head'], snapshot_sha256=sha(SNAPSHOT),
                  protected_file_count=len(state['files']), protected_bytes=state['protected_bytes'],
                  changed_files=changes,
                  online_occupancy_sha256=sha(ROOT / 'lift_curtain/perception/online_occupancy.py'))
    (REPORT / 'PRESERVATION_AUDIT.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if changes:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
