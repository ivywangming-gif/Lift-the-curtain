"""Fail-closed streaming validation of recorded V5 discrete 200 Hz history.

This is evidence validation, not a collision test or a continuous-time claim.
The caller must audit ``initial_arrays`` as well as every yielded poststep chunk.
``summary`` becomes VERIFIED_HISTORY only after the iterator is fully consumed.
Legacy single-NPZ 50 Hz evidence is deliberately rejected by this V5 interface.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np

from collision_geometry import GeometryNotVerified

DT = .005
CLOCK_TOLERANCE_S = 1e-6
# SimulationContext._current_time += callback step_size. Native callbacks can
# expose float32(.005), while some builds retain nominal double precision.
# These are two preregistered constants, never a slope fitted to a recording.
ACTUAL_CLOCK_STEP_CANDIDATES_S = np.array([DT, float(np.float32(DT))])
RECOVERY_PHASES = {"responsive_release_stand", "physical_post_retract_stand"}


def _require(condition, reason):
    if not condition:
        raise GeometryNotVerified(reason)


def _integer(value, name, minimum=0):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value >= minimum,
             "INVALID_INTEGER:" + name)
    return int(value)


def _close(actual, expected, reason):
    _require(np.isfinite(actual) and np.isfinite(expected) and abs(float(actual)-float(expected)) <= CLOCK_TOLERANCE_S, reason)


class VerifiedChunkHistory:
    def __init__(self, run, capture, geom):
        self.run = Path(run).resolve()
        self.capture, self.geom = capture, geom
        self.summary = dict(status="NOT_FULLY_CONSUMED", source_files={}, sample_count=0,
                            history_layout="CHUNKED_NPZ_V2",
                            scope="Every recorded discrete 200 Hz state from initialization; no continuous-time claim")
        self._started = False
        capture_path = self.run/"g_body_collision_capture.json"
        _require(capture_path.is_file() and json.loads(capture_path.read_text(encoding="utf-8-sig")) == capture,
                 "CAPTURE_ARGUMENT_NOT_BOUND_TO_RECORDED_FILE")
        self.summary["source_files"][capture_path.name] = hashlib.sha256(capture_path.read_bytes()).hexdigest()
        geometry_path = self._verified_path(capture["geometry"]["path"], capture["geometry"]["sha256"])
        _require(json.loads(geometry_path.read_text(encoding="utf-8-sig")) == geom, "GEOMETRY_ARGUMENT_NOT_BOUND_TO_CAPTURE")
        _require(capture.get("history_layout") == "CHUNKED_NPZ_V2" and capture.get("nominal_hz") == 200,
                 "V5_200HZ_CHUNK_HISTORY_REQUIRED_LEGACY_NOT_ACCEPTED")
        _require(capture.get("episode_finalized") is True and capture.get("synchronized_poststep") is True,
                 "HISTORY_NOT_FINALIZED_SYNCHRONIZED")
        _require(capture.get("pose_source") == "body_link_pose_w" and capture.get("quaternion_order") == "WXYZ",
                 "BODY_LINK_FRAME_OR_QUATERNION_ORDER_UNVERIFIED")
        index_path = self._verified_path(capture["history_index_path"], capture["history_index_sha256"])
        self.index = json.loads(index_path.read_text(encoding="utf-8-sig"))
        _require(self.index.get("schema") == "round4070-g.body_collision_history_chunks.v2"
                 and self.index.get("nominal_hz") == 200 and self.index.get("episode_finalized") is True
                 and self.index.get("model_input") is False, "INVALID_V5_HISTORY_INDEX")
        self.names = self.index["body_names"]
        _require(isinstance(self.names, list) and self.names and len(set(self.names)) == len(self.names)
                 and self.names == geom["body_names"] and capture.get("body_count") == len(self.names), "BODY_NAMES_OR_COUNT_MISMATCH")
        self.node_count = _integer(self.index["node_count"], "node_count")
        expected_nodes = 0
        for sheet in geom["sheets"]:
            _require(sheet["global_node_offset"] == expected_nodes, "SHEET_NODE_OFFSETS_NOT_COMPLETE_CONTIGUOUS")
            expected_nodes += _integer(sheet["vertex_count"], "sheet vertex_count")
        _require(self.node_count == expected_nodes, "GEOMETRY_NODE_COUNT_MISMATCH")
        self.total = _integer(self.index["sample_count"], "sample_count", 1)
        _require(capture.get("sample_count") == self.total, "CAPTURE_INDEX_SAMPLE_COUNT_MISMATCH")
        self.anchors = capture["execution_anchors"]
        for name in ("last_physics_step", "executed_step_count", "independently_expected_total_steps"):
            _require(_integer(self.anchors[name], name, 1) == self.total, "ABSOLUTE_EXECUTION_ANCHOR_MISMATCH:" + name)
        _require(self.anchors["first_physics_step"] == 1, "FIRST_ACTUAL_STEP_NOT_ONE")
        _close(self.anchors["first_time_s"], DT, "FIRST_TIME_ANCHOR_MISMATCH")
        _close(self.anchors["final_time_s"], self.total*DT, "FINAL_TIME_ANCHOR_MISMATCH")
        self.initial_time = float(self.anchors["initial_actual_simulation_time_s"])
        self._clock_candidates = np.ones(len(ACTUAL_CLOCK_STEP_CANDIDATES_S), dtype=bool)
        self._check_actual_clock(np.array([self.total]), np.array([self.anchors["final_actual_simulation_time_s"]],float))
        initial = capture["initial_state"]
        _require(initial.get("physics_step") == 0 and initial.get("stage") == "AFTER_RESET_AND_RNG_BEFORE_FIRST_ACTUAL_STEP",
                 "INITIAL_STATE_NOT_RESET_STEP_ZERO")
        _close(initial["actual_simulation_time_s"], self.initial_time, "INITIAL_STATE_CLOCK_MISMATCH")
        initial_path = self._verified_path(initial["path"], initial["sha256"])
        with np.load(initial_path, allow_pickle=False) as data:
            self.initial_arrays = {k: data[k] for k in data.files}
        initial_arrays = self.initial_arrays
        _require(np.asarray(initial_arrays["physics_step"]).shape == () and int(initial_arrays["physics_step"]) == 0,
                 "INITIAL_ARRAY_STEP_NOT_ZERO")
        _close(float(initial_arrays["actual_simulation_time_s"]), self.initial_time, "INITIAL_ARRAY_CLOCK_MISMATCH")
        self._finite_shape(initial_arrays["body_link_pose_w"], (len(self.names), 7), "initial body pose")
        self._unit_quaternions(initial_arrays["body_link_pose_w"][:, 3:])
        for name in ("nodes_world", "node_velocities_world"):
            self._finite_shape(initial_arrays[name], (self.node_count, 3), "initial " + name)
        self.ledger = capture["phase_execution_ledger"]
        _require(isinstance(self.ledger, list) and self.ledger, "MISSING_PHASE_EXECUTION_LEDGER")
        self.active = []
        expected_step = 1
        for identity, entry in enumerate(self.ledger):
            _require(entry.get("phase_instance_id") == identity and isinstance(entry.get("phase"), str), "PHASE_INSTANCE_ID_MISMATCH")
            count = _integer(entry["executed_step_count"], "phase executed_step_count")
            if count == 0:
                _require(entry.get("call_status") == "ZERO_STEPS" and all(entry.get(k) is None for k in
                    ("start_physics_step", "end_physics_step", "start_time_s", "end_time_s", "final_snapshot_physics_step")),
                    "ZERO_STEP_PHASE_HAS_FAKE_SAMPLE_OR_FINAL")
                continue
            _require(entry.get("call_status") == "COMPLETED", "INCOMPLETE_OR_FAILED_PHASE_CALL")
            _require(entry.get("start_physics_step") == expected_step and entry.get("end_physics_step") == expected_step+count-1,
                     "PHASE_LEDGER_STEP_GAP_OR_OVERLAP")
            _require(entry.get("final_snapshot_physics_step") == entry["end_physics_step"], "PHASE_FINAL_SNAPSHOT_MISSING")
            _close(entry["start_time_s"], expected_step*DT, "PHASE_START_TIME_MISMATCH")
            _close(entry["end_time_s"], (expected_step+count-1)*DT, "PHASE_END_TIME_MISMATCH")
            expected_step += count
            self.active.append(entry)
        _require(expected_step == self.total+1, "PHASE_LEDGER_TOTAL_MISMATCH")
        _require(self.active and self.active[0]["phase"] == "initialization_physics_step", "INITIALIZATION_STEP_PHASE_MISSING")
        _require(any(e["phase"].startswith("physical_probe") for e in self.active), "ACTUAL_PROBE_PHASE_MISSING")
        _require(self.active[-1]["phase"] in RECOVERY_PHASES, "FINAL_ACTUAL_PHASE_IS_NOT_RECOVERY")
        _require(set(capture["phase_names"]) == {e["phase"] for e in self.active}, "CAPTURE_PHASE_NAMES_MISMATCH")
        self.chunks = self.index["chunks"]
        _require(isinstance(self.chunks, list) and self.chunks, "NO_HISTORY_CHUNKS")
        _require(len({c["path"] for c in self.chunks}) == len(self.chunks), "DUPLICATE_CHUNK_PATH")
        self.seen_phase = {e["phase_instance_id"]: 0 for e in self.active}
        self.boundaries = set()

    def _verified_path(self, name, digest):
        relative = Path(name)
        _require(not relative.is_absolute() and ".." not in relative.parts, "EVIDENCE_PATH_ESCAPES_RUN")
        path = (self.run/relative).resolve()
        _require(path.is_relative_to(self.run) and path.is_file(), "MISSING_OR_EXTERNAL_EVIDENCE_FILE")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(isinstance(digest, str) and actual == digest, "EVIDENCE_SHA256_MISMATCH:" + str(relative))
        self.summary["source_files"][relative.as_posix()] = actual
        return path

    @staticmethod
    def _finite_shape(value, shape, name):
        array = np.asarray(value)
        _require(array.shape == shape and array.dtype.kind in "fiu" and np.isfinite(array).all(), "INVALID_ARRAY:" + name)

    @staticmethod
    def _unit_quaternions(quats):
        _require(np.all(np.abs(np.linalg.norm(quats, axis=-1)-1.) <= 1e-4), "NON_UNIT_BODY_QUATERNION")

    def _check_actual_clock(self, steps, values):
        _require(np.isfinite(values).all() and np.isfinite(self.initial_time), "NONFINITE_ACTUAL_CLOCK")
        expected = self.initial_time + np.asarray(steps)[:,None]*ACTUAL_CLOCK_STEP_CANDIDATES_S[None,:]
        self._clock_candidates &= np.all(np.abs(np.asarray(values)[:,None]-expected) <= CLOCK_TOLERANCE_S,axis=0)
        _require(bool(self._clock_candidates.any()), "ACTUAL_200HZ_CLOCK_MISMATCH_NO_SINGLE_PREDEFINED_DT_MATCHES")

    def __iter__(self):
        _require(not self._started, "HISTORY_ITERATOR_IS_SINGLE_PASS")
        self._started = True
        count = 0
        last_boundary = False
        for chunk in self.chunks:
            path = self._verified_path(chunk["path"], chunk["sha256"])
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            n = _integer(chunk["sample_count"], "chunk sample_count", 1)
            _require(n <= 500, "CHUNK_EXCEEDS_FROZEN_500_STEP_LIMIT")
            for name in ("time_s", "actual_simulation_time_s", "physics_step", "phase_instance_id", "phase", "boundary"):
                _require(np.asarray(arrays[name]).shape == (n,), "INVALID_VECTOR_SHAPE:" + name)
            for name in ("physics_step", "phase_instance_id"):
                _require(arrays[name].dtype.kind in "iu", "STEP_OR_PHASE_ID_NOT_INTEGER")
            _require(arrays["boundary"].dtype.kind == "b" and arrays["phase"].dtype.kind in "US", "PHASE_OR_BOUNDARY_DTYPE_INVALID")
            expected = np.arange(count+1, count+n+1)
            _require(np.array_equal(arrays["physics_step"], expected), "MISSING_DUPLICATE_OR_TRUNCATED_ACTUAL_STEP")
            for name, expected_time in (("time_s", expected*DT),):
                self._finite_shape(arrays[name], (n,), name)
                _require(np.all(np.abs(arrays[name]-expected_time) <= CLOCK_TOLERANCE_S), "ACTUAL_200HZ_CLOCK_MISMATCH:" + name)
            self._check_actual_clock(expected, arrays["actual_simulation_time_s"])
            _require(chunk["first_physics_step"] == int(expected[0]) and chunk["last_physics_step"] == int(expected[-1]), "CHUNK_STEP_METADATA_MISMATCH")
            _close(chunk["first_time_s"], expected[0]*DT, "CHUNK_FIRST_TIME_MISMATCH")
            _close(chunk["last_time_s"], expected[-1]*DT, "CHUNK_LAST_TIME_MISMATCH")
            for name, shape in (("body_positions_world", (n,len(self.names),3)), ("body_quaternions_wxyz", (n,len(self.names),4)),
                                ("nodes_world", (n,self.node_count,3)), ("node_velocities_world", (n,self.node_count,3))):
                self._finite_shape(arrays[name], shape, name)
            self._unit_quaternions(arrays["body_quaternions_wxyz"])
            for offset, (step, identity, phase, boundary) in enumerate(zip(expected, arrays["phase_instance_id"], arrays["phase"], arrays["boundary"])):
                identity = int(identity)
                _require(0 <= identity < len(self.ledger), "UNREGISTERED_PHASE_ID")
                entry = self.ledger[identity]
                _require(entry["executed_step_count"] > 0 and entry["phase"] == str(phase)
                         and entry["start_physics_step"] <= step <= entry["end_physics_step"], "SAMPLE_PHASE_DOES_NOT_OWN_ACTUAL_STEP")
                self.seen_phase[identity] += 1
                if boundary:
                    _require(step == entry["end_physics_step"], "BOUNDARY_IS_NOT_ACTUAL_PHASE_FINAL_STEP")
                    self.boundaries.add((identity, int(step)))
            count += n
            _require(count <= self.total, "EXCESS_HISTORY_SAMPLES")
            self.summary["sample_count"] = count
            last_boundary = bool(arrays["boundary"][-1])
            yield arrays
        _require(count == self.total, "TRUNCATED_HISTORY_SUFFIX")
        _require(all(self.seen_phase[e["phase_instance_id"]] == e["executed_step_count"] for e in self.active), "PHASE_SAMPLE_TOTAL_MISMATCH")
        final = self.active[-1]
        _require(last_boundary and (final["phase_instance_id"], self.total) in self.boundaries, "FINAL_RECOVERY_BOUNDARY_NOT_ACTUALLY_RECORDED")
        for request in self.capture.get("phase_boundary_requests", []):
            if request.get("recorded") is True:
                identity = _integer(request["phase_instance_id"], "boundary request phase ID")
                _require(identity < len(self.ledger) and self.ledger[identity]["phase"] == request["phase"]
                         and (identity, request["requested_physics_step"]) in self.boundaries, "BOUNDARY_REQUEST_HAS_NO_ACTUAL_FINAL_SAMPLE")
        self.summary.update(status="VERIFIED_HISTORY", phase_ledger_verified=True, execution_anchors_verified=True,
                            initial_state_verified=True, initial_state_collision_test_required=True, sample_count=count,
                            phase_count=len(self.ledger), zero_step_phase_count=len(self.ledger)-len(self.active),
                            actual_clock_matching_fixed_dt_s=ACTUAL_CLOCK_STEP_CANDIDATES_S[self._clock_candidates].tolist(),
                            actual_clock_contract="one unchanged predefined nominal-double or native-float32(.005) callback dt across all states; no fitted time slope; absolute tolerance 1e-6 s",
                            first_physics_step=1,last_physics_step=self.total,nominal_hz=200)


def iterate_verified_chunks(run, capture, geom):
    return VerifiedChunkHistory(run, capture, geom)


def verify_history(run, capture, geom):
    """Validate all history bytes and return a ledger summary, not collision validity."""
    history = iterate_verified_chunks(run, capture, geom)
    for _ in history:
        pass
    return history.summary
