"""Read-only support-bound diagnostic; never changes a formal collision audit.

The full native vertex hull remains an OUTER envelope. Its support function is
an upper bound for the actual solid, not an assertion of exact native geometry.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from collision_clearance import _rotations
from physics_history import iterate_verified_chunks
from physics_wall_evidence import verified_wall_for_audit


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose(run):
    run = Path(run).resolve()
    geometry = read(run / "g_robot_collision_geometry.json")
    native = read(run / "g_native_geometry_verified.json")
    capture = read(run / "g_body_collision_capture.json")
    sources = {name: digest(run / name) for name in [
        "g_robot_collision_geometry.json", "g_native_geometry_verified.json",
        "g_body_collision_capture.json", "g_collision_audit.json"]}
    wall = verified_wall_for_audit(run, geometry, sources)
    if wall is None:
        raise ValueError("Diagnostic requires a verified actual static wall")
    center = np.array(wall["center_xyz"])
    size = np.array(wall["size_xyz"])
    low, high = center - size / 2, center + size / 2
    if native["body_names"] != geometry["body_names"]:
        raise ValueError("Body order differs")
    selected = [(i, p) for i, p in enumerate(native["parts"])
                if p.get("exact_intersection_allowed") is False]
    if not selected or any(p["type"] != "convex" for _, p in selected):
        raise ValueError("Unexpected conservative geometry")
    reports = []
    for index, part in selected:
        reports.append(dict(part_index=index, body_index=part["body_index"],
            body_name=part["body_name"], evidence_mode=part["evidence_mode"],
            maximum_front_plane_support_x_m=-float("inf"),
            front_plane_crossing_state_count=0,
            eroded_front_plane_crossing_state_count=0,
            overlapping_world_aabb_state_count=0,
            suspect_states=[]))
    history = iterate_verified_chunks(run, capture, geometry)
    count = 0

    def consume(steps, times, phases, positions, quaternions):
        nonlocal count
        for j, step in enumerate(steps):
            rotations, _ = _rotations(quaternions[j], 1e-4)
            count += 1
            for (_, part), report in zip(selected, reports):
                body = part["body_index"]
                vertices = np.asarray(part["vertices_body_m"], dtype=float)
                world = vertices @ rotations[body].T + positions[j, body]
                minimum, maximum = world.min(0), world.max(0)
                support = float(maximum[0])
                sample = dict(physics_step=int(step), time_s=float(times[j]), phase=str(phases[j]),
                    world_bounds_m=[minimum.tolist(), maximum.tolist()],
                    front_plane_overlap_upper_bound_m=support-float(low[0]),
                    separation_from_5mm_eroded_wall_front_m=float(low[0])+.005-support,
                    support_vertex_world_m=world[int(np.argmax(world[:, 0]))].tolist())
                if support > report["maximum_front_plane_support_x_m"]:
                    report["maximum_front_plane_support_x_m"] = support
                    report["maximum_support_state"] = sample
                report["front_plane_crossing_state_count"] += int(support > low[0])
                report["eroded_front_plane_crossing_state_count"] += int(support > low[0]+.005)
                overlaps = bool(np.all(maximum >= low) and np.all(minimum <= high))
                report["overlapping_world_aabb_state_count"] += int(overlaps)
                if overlaps:
                    report["suspect_states"].append(sample)

    initial = history.initial_arrays
    pose = initial["body_link_pose_w"]
    consume([0], [0.], ["INITIAL_RESET_STATE"], pose[None, :, :3], pose[None, :, 3:])
    for arrays in history:
        consume(arrays["physics_step"], arrays["time_s"], arrays["phase"],
                arrays["body_positions_world"], arrays["body_quaternions_wxyz"])
    if history.summary["status"] != "VERIFIED_HISTORY":
        raise ValueError("Incomplete history")
    sources.update(history.summary["source_files"])
    for report in reports:
        report["maximum_front_plane_overlap_upper_bound_m"] = report["maximum_front_plane_support_x_m"] - float(low[0])
        report["minimum_5mm_eroded_front_separation_m"] = float(low[0])+.005-report["maximum_front_plane_support_x_m"]
        report["support_separation_by_tolerance_m"] = {
            str(tol): dict(
                maximum_outer_support_plus_numerical_margin_m=report["maximum_front_plane_support_x_m"]+1e-6,
                eroded_wall_front_m=float(low[0])+tol,
                minimum_support_separation_margin_m=float(low[0])+tol-report["maximum_front_plane_support_x_m"]-1e-6,
                sufficiently_separated=bool(report["maximum_front_plane_support_x_m"]+1e-6 < float(low[0])+tol))
            for tol in [0., .001, .005]}
    return dict(schema="round4070-g.conservative_wall_support_diagnostic.v2",
        purpose="Read-only diagnostic, not an admission decision or replacement formal audit",
        formal_audit_unchanged=True, formal_audit_status=read(run/"g_collision_audit.json")["status"],
        actual_geometry_remains_conservative=True, continuous_time_claim=False,
        original_run=str(run), wall=wall, wall_bounds_m=[low.tolist(), high.tolist()],
        strict_penetration_tolerance_m=.005, source_files=sources,
        numerical_support_margin_m=1e-6,
        numerical_margin_scope="Deterministic floating-point protection, not calibrated physical or measurement uncertainty",
        support_definition="Maximum world-x over every complete native vertex; exactly the support of their convex hull, used only as an outer bound",
        diagnostic_source_sha256=digest(__file__), verified_history=history.summary,
        audited_state_count=count, conservative_part_count=len(reports), parts=reports,
        all_outer_envelopes_below_5mm_eroded_wall_front=all(
            p["minimum_5mm_eroded_front_separation_m"] > 1e-6 for p in reports),
        mathematical_scope=("For actual solid A contained in outer O, max_x(A)<=max_x(O). "
            "If max_x(O)<wall_min_x+5mm, A cannot reach the wall's 5mm-eroded interior. "
            "Also erosion(O,5mm) has max_x<=max_x(O)-5mm, so an actual wall point "
            "at x>=wall_min_x cannot be 5mm inside O or A. This is a sufficient "
            "support-plane separation argument with the displayed positive margin; "
            "the frozen any-envelope-overlap NOT_VERIFIED admission rule is unchanged."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = diagnose(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(state_count=result["audited_state_count"],
        conservative_parts=result["conservative_part_count"],
        separated=result["all_outer_envelopes_below_5mm_eroded_wall_front"],
        maximum_overlap_m=max(p["maximum_front_plane_overlap_upper_bound_m"] for p in result["parts"]),
        output=str(args.output))))


if __name__ == "__main__":
    main()
