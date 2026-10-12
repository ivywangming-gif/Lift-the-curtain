"""No-render smoke test for the recording request contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lift_curtain.recording import (
    RecordingSpec,
    classify_deformable_visual_sync,
    parse_resolution,
    write_recording_request,
)


def main() -> int:
    assert parse_resolution("320x240") == (320, 240)
    assert parse_resolution("320,240") == (320, 240)
    assert classify_deformable_visual_sync(0.02, 0.0) == "VISUAL_DEFORMABLE_MISMATCH"
    root = REPO / "reports" / "release"
    request = root / ".round17_recording_request.json"
    try:
        write_recording_request(request, RecordingSpec("curtain", root))
        payload = json.loads(request.read_text(encoding="utf-8"))
        assert payload["verified_video"] is False
        assert payload["status"] == "REQUESTED_NOT_STARTED"
    finally:
        request.unlink(missing_ok=True)
    print("RECORDING_STATIC_SMOKE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
