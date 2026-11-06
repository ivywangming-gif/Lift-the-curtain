"""Durable G collection -> audited snapshot -> eligible learning orchestration.

No stage is called scientifically complete merely because its process returned.
The collection queue owns physics validity, retry limits and the 1000 launch cap.
This wrapper audits each bounded collection chunk and never changes old arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from resource_lock import resource_lock
from simulation_process import is_simulation_process

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'
DATA_FILES = ('manifest.json', 'split_manifest.json', 'matched_pairs.json', 'dataset_status.json')
LEARNING_FILES = ('train_benchmark.py', 'learning_core.py', 'learning_models.py',
                  'engineering_rules.py', 'observation_randomization.py', 'resource_lock.py',
                  'physics_history.py', 'collision_geometry.py', 'g1_action_contract_v2.py',
                  'simulation_process.py')


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return default


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def event(kind, **details):
    row = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(), event=kind, **details)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'workflow_events.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n'); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps(row, ensure_ascii=False), flush=True)


def call(script, *arguments):
    argv = [sys.executable, str(SCRIPTS / script), *map(str, arguments)]
    event('STAGE_STARTED', script=script, arguments=list(map(str, arguments)))
    # The guarded collection child exclusively owns and cleans up its simulator.
    result = subprocess.run(argv, cwd=ROOT)
    event('STAGE_RETURNED', script=script, exit_code=result.returncode)
    if result.returncode:
        raise RuntimeError(f'{script} returned {result.returncode}; preserve logs and repair before resuming')


def launch_count():
    return sum(1 for kind in ('physics', 'trials') for _ in (OUT / kind).glob('*/*/campaign_launch.json'))


def current_manifest():
    pointer = read(DOC / 'CURRENT_CAMPAIGN.json')
    if not pointer or Path(pointer['manifest']).name != pointer['manifest']:
        raise RuntimeError('Missing or invalid frozen campaign pointer')
    path = OUT / pointer['manifest']
    if not path.exists():
        call('restore_campaign.py')
    if sha(path) != pointer['sha256']:
        raise RuntimeError('Current campaign SHA256 mismatch')
    return path


def snapshot_dataset():
    """Copy immutable metadata; NPZ remain in the read-only, hash-checked exporter."""
    hashes = {name: sha(OUT / 'dataset' / name) for name in DATA_FILES}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    destination = OUT / 'dataset_snapshots' / identity
    destination.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        source = OUT / 'dataset' / name
        target = destination / name
        if target.exists():
            if sha(target) != hashes[name]:
                raise RuntimeError(f'Immutable snapshot changed: {target}')
        else:
            with target.open('xb') as handle:
                handle.write(source.read_bytes()); handle.flush(); os.fsync(handle.fileno())
        if sha(target) != hashes[name]:
            raise RuntimeError('Dataset changed while snapshotting; reject mixed snapshot')
    index = destination / 'snapshot.json'
    if not index.exists():
        atomic(index, dict(schema='round4070-g.dataset_snapshot.v1', identity=identity, files=hashes,
            npz_policy='Repository-relative original NPZ, hashes validated by training loader; no resampling or copies',
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    return destination, identity


def audit():
    call('build_dataset.py')
    from physics_campaign import summarize
    summary = summarize(read(current_manifest()))
    event('SCENE_COVERAGE_REFRESHED', named_rows=len(summary['scene_states']), actual_g1_launches=summary['g1_launches'])
    call('checkpoint.py')
    call('report_status.py')
    event('ACTUAL_DATA_AUDITED', **read(OUT / 'dataset/dataset_status.json', {}))


def learning_source_hashes():
    """Use the trainer's freeze contract, failing closed on registry drift."""
    from train_benchmark import sources
    code = sources()
    if set(code) != set(LEARNING_FILES):
        raise RuntimeError('Workflow learning-source registry differs from trainer freeze contract')
    if code != {name: sha(SCRIPTS / name) for name in LEARNING_FILES}:
        raise RuntimeError('Learning source changed while computing orchestration identity')
    return code


def freeze_learning(snapshot, identity):
    code = learning_source_hashes()
    protocol_id = hashlib.sha256(json.dumps(dict(dataset=identity, code=code), sort_keys=True).encode()).hexdigest()
    run = OUT / ('learning_campaign_' + protocol_id[:20])
    if not (run / 'frozen_protocol.json').exists():
        call('train_benchmark.py', 'freeze', '--manifest', snapshot / 'manifest.json',
             '--splits', snapshot / 'split_manifest.json', '--pairs', snapshot / 'matched_pairs.json', '--output', run)
    # Verify both newly created and reused protocols. A source edit during the
    # child freeze must not publish a run under an identity computed earlier.
    protocol = read(run / 'frozen_protocol.json')
    if (protocol['manifest_sha256'] != sha(snapshot / 'manifest.json') or protocol['code_sha256'] != code
        or learning_source_hashes() != code):
        raise RuntimeError('Frozen learning protocol mismatch; never overwrite a prior protocol')
    if not (run / 'task_queue.json').exists():
        raise RuntimeError('Interrupted freeze is incomplete; inspect this run before creating a new one')
    atomic(OUT / 'latest_learning_run.json', dict(run=str(run.relative_to(ROOT)), dataset_snapshot=identity,
        scientific_status='IN_PROGRESS_NOT_COMPLETE'))
    return run


def active_simulations():
    active = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            args = process.info['cmdline'] or []
            if is_simulation_process(process.info['name'], args):
                active.append(process.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return active


def train(run, max_fits=None, hours=24.):
    if (OUT / 'PAUSE_G1.flag').exists():
        event('TRAINING_WITHHELD_G1_ADMISSION_PENDING', run=str(run))
        return
    queue = read(run / 'task_queue.json')
    eligible = [task for task in queue['tasks'] if task['status'] in ('PLANNED', 'TRAINED', 'FAILED_WITH_REASON')]
    if not eligible:
        event('NO_ELIGIBLE_UNFINISHED_FITS', run=str(run), counts=dict(Counter(t['status'] for t in queue['tasks'])))
        return
    window_path = OUT / 'training_window.json'
    old = read(window_path, {})
    if old.get('allow_training') and dt.datetime.fromisoformat(old['expires_utc'].replace('Z', '+00:00')) > dt.datetime.now(dt.timezone.utc):
        raise RuntimeError('Another active training window exists; do not replace its owner')
    if active_simulations() or (OUT / 'resource_lock.json').exists() or (OUT / 'campaign_lock.json').exists():
        raise RuntimeError('Training requires completed collection boundary with no active simulation or campaign lock')
    token = uuid.uuid4().hex
    window = dict(allow_training=True, isaac_stopped_confirmed=True, owner_pid=os.getpid(), owner_token=token,
        expires_utc=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)).isoformat(),
        reason='Actual collection child returned; no campaign/resource lock or simulation process; trainer rechecks under atomic lock')
    atomic(window_path, window)
    try:
        arguments = ['run', '--output', run, '--training-window', window_path, '--device', 'cuda']
        if max_fits is not None:
            arguments.extend(['--max-fits', max_fits])
        call('train_benchmark.py', *arguments)
    finally:
        current = read(window_path, {})
        if current.get('owner_token') == token:
            atomic(window_path, {**current, 'allow_training': False, 'closed_utc': dt.datetime.now(dt.timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'analyze', 'train'], default='all')
    parser.add_argument('--chunk-jobs', type=int, default=12)
    parser.add_argument('--max-jobs', type=int, default=0, help='Total collection jobs this invocation; zero means remaining eligible queue')
    parser.add_argument('--scene')
    parser.add_argument('--retry-runtime-failures', action='store_true')
    parser.add_argument('--max-fits', type=int)
    parser.add_argument('--training-hours', type=float, default=24.)
    args = parser.parse_args()
    if args.chunk_jobs < 1 or args.max_jobs < 0 or args.training_hours <= 0:
        parser.error('Positive chunk size/training hours and nonnegative job limit required')
    with resource_lock('workflow', f'workflow-{os.getpid()}', OUT / 'workflow_lock.json'):
        manifest = current_manifest()
        initial = launch_count()
        collection = 'NOT_REQUESTED'
        if args.mode == 'all':
            while True:
                if current_manifest()!=manifest:
                    raise RuntimeError('Current frozen campaign changed; checkpoint and restart under the new explicit contract')
                before = launch_count()
                remaining = args.max_jobs - (before - initial) if args.max_jobs else args.chunk_jobs
                if remaining <= 0:
                    collection = 'INVOCATION_LIMIT_CHECKPOINTED'; break
                extra = ['--manifest', manifest, '--mode', 'all', '--max-jobs', min(args.chunk_jobs, remaining)]
                if args.scene:
                    extra.extend(['--scene', args.scene])
                if args.retry_runtime_failures:
                    extra.append('--retry-runtime-failures')
                events_file = OUT / 'campaign_events.jsonl'
                cursor = events_file.stat().st_size if events_file.exists() else 0
                call('physics_campaign.py', *extra)
                audit()
                with events_file.open('rb') as handle:
                    handle.seek(cursor)
                    events = [json.loads(line) for line in handle if line.strip()]
                if any(row['event'] == 'GLOBAL_IMPLEMENTATION_ERROR_PAUSE' for row in events):
                    collection = 'BLOCKED_WITH_REASON_GLOBAL_IMPLEMENTATION_ERROR'; break
                if launch_count() == before:
                    if (OUT / 'PAUSE_G1.flag').exists() or any(row['event'] == 'G1_ADMISSION_PAUSED' for row in events):
                        event('WAIT_G1_ADMISSION', training_permitted=False, physics_queue_currently_exhausted=True)
                        atomic(OUT / 'workflow_wait_checkpoint.json', dict(status='IN_PROGRESS_NOT_COMPLETE', reason='G1_ADMISSION_PAUSED', training_permitted=False, pid=os.getpid()))
                        while (OUT / 'PAUSE_G1.flag').exists():
                            time.sleep(15)
                        continue
                    collection = 'NO_REMAINING_ELIGIBLE_COLLECTION_JOBS'; break
        audit()
        snapshot, identity = snapshot_dataset()
        run = freeze_learning(snapshot, identity)
        if args.mode != 'analyze' and not collection.startswith('BLOCKED'):
            train(run, args.max_fits, args.training_hours)
        call('learning_report.py', '--run', run)
        call('checkpoint.py')
        call('report_status.py')
        result = dict(status='IN_PROGRESS_NOT_COMPLETE', collection_status=collection,
            additional_actual_launches=launch_count()-initial, dataset_snapshot=str(snapshot.relative_to(ROOT)),
            learning_run=str(run.relative_to(ROOT)),
            learning_counts=dict(Counter(t['status'] for t in read(run / 'task_queue.json')['tasks'])),
            checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            warning='Process completion is not scientific completion; inspect physics, label, split and per-method limitations')
        atomic(OUT / 'workflow_checkpoint.json', result)
        event('WORKFLOW_CHECKPOINTED', **result)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        event('WORKFLOW_INTERRUPTED', exception=type(exc).__name__, reason=str(exc),
              status='IN_PROGRESS_NOT_COMPLETE', resume='powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_round4070_g.ps1 -Mode all -RetryRuntimeFailures')
        raise
