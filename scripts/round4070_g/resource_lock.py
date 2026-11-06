"""Atomic mutual exclusion shared by G simulation and full learning runs.

An abandoned lock is never guessed away. The coordinator must establish that
the recorded PID has exited before archiving/removing that specific lock.
"""
from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "outputs/round4070_comprehensive_traversability/resource_lock.json"


@contextmanager
def resource_lock(owner, run_id, path=LOCK_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    record = dict(owner=owner, run_id=run_id, pid=os.getpid(), token=token,
                  created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = "unreadable existing lock"
        raise InterruptedError("RESOURCE_LOCK_BUSY: " + existing) from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield record
    finally:
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current.get("token") == token and current.get("pid") == os.getpid():
                path.unlink()
        except FileNotFoundError:
            pass
