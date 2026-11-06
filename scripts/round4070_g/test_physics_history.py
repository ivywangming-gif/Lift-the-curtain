"""Synthetic byte/hash/clock/phase counterexamples; no actual physics claim."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from physics_history import iterate_verified_chunks, verify_history, VerifiedChunkHistory
from collision_geometry import GeometryNotVerified

TEMP_ROOT=Path(__file__).resolve().parents[2]/"outputs/round4070_comprehensive_traversability/implementation_smoke/history_contracts"
TEMP_ROOT.mkdir(parents=True,exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(run, total=503):
    names=["body"]
    geom=dict(body_names=names,sheets=[dict(global_node_offset=0,vertex_count=3)])
    geometry=run/"geometry.json";write(geometry,geom)
    initial=run/"initial.npz"
    np.savez(initial,body_link_pose_w=np.array([[0,0,0,1,0,0,0.]]),nodes_world=np.zeros((3,3)),node_velocities_world=np.zeros((3,3)),actual_simulation_time_s=.01,physics_step=0)
    ledger=[]
    for identity,(phase,start,end) in enumerate((("initialization_physics_step",1,1),("physical_probe",2,total-2),("unused_zero",None,None),("responsive_release_stand",total-1,total))):
        count=0 if start is None else end-start+1
        ledger.append(dict(phase_instance_id=identity,phase=phase,start_physics_step=start,end_physics_step=end,executed_step_count=count,
                           final_snapshot_physics_step=end,start_time_s=None if start is None else start*.005,end_time_s=None if end is None else end*.005,call_status="COMPLETED" if count else "ZERO_STEPS"))
    chunks=[]
    for start in range(1,total+1,500):
        steps=np.arange(start,min(start+500,total+1));n=len(steps)
        ids=np.where(steps==1,0,np.where(steps<total-1,1,3));phases=np.array([ledger[i]["phase"] for i in ids])
        arrays=dict(physics_step=steps,time_s=steps*.005,actual_simulation_time_s=.01+steps*.005,phase_instance_id=ids,phase=phases,boundary=steps==total,
                    body_positions_world=np.zeros((n,1,3)),body_quaternions_wxyz=np.tile([1.,0,0,0],(n,1,1)),nodes_world=np.zeros((n,3,3)),node_velocities_world=np.zeros((n,3,3)))
        path=run/f"chunk_{len(chunks)}.npz";np.savez(path,**arrays)
        chunks.append(dict(path=path.name,sha256=sha(path),sample_count=n,first_physics_step=int(steps[0]),last_physics_step=int(steps[-1]),first_time_s=float(steps[0]*.005),last_time_s=float(steps[-1]*.005)))
    index=dict(schema="round4070-g.body_collision_history_chunks.v2",nominal_hz=200,sample_count=total,body_names=names,node_count=3,chunks=chunks,episode_finalized=True,model_input=False)
    index_path=run/"index.json";write(index_path,index)
    capture=dict(history_layout="CHUNKED_NPZ_V2",nominal_hz=200,episode_finalized=True,synchronized_poststep=True,pose_source="body_link_pose_w",quaternion_order="WXYZ",
                 history_index_path=index_path.name,history_index_sha256=sha(index_path),geometry=dict(path=geometry.name,sha256=sha(geometry)),sample_count=total,body_count=1,
                 phase_names=[e["phase"] for e in ledger if e["executed_step_count"]],phase_execution_ledger=ledger,
                 initial_state=dict(path=initial.name,sha256=sha(initial),physics_step=0,actual_simulation_time_s=.01,stage="AFTER_RESET_AND_RNG_BEFORE_FIRST_ACTUAL_STEP"),
                 execution_anchors=dict(first_physics_step=1,last_physics_step=total,executed_step_count=total,first_time_s=.005,final_time_s=total*.005,
                                        initial_actual_simulation_time_s=.01,final_actual_simulation_time_s=.01+total*.005,independently_expected_total_steps=total),
                 phase_boundary_requests=[dict(phase="responsive_release_stand",phase_instance_id=3,requested_physics_step=total,recorded=True)])
    write(run/"g_body_collision_capture.json",capture)
    return capture,geom,index


def refresh(run,capture,index):
    for chunk in index["chunks"]:chunk["sha256"]=sha(run/chunk["path"])
    write(run/"index.json",index);capture["history_index_sha256"]=sha(run/"index.json")
    write(run/"g_body_collision_capture.json",capture)


def change_chunk(run,index,number,change):
    path=run/index["chunks"][number]["path"]
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    change(arrays);np.savez(path,**arrays)


class HistoryContracts(unittest.TestCase):
    def test_long_native_float32_callback_clock_and_fixed_candidate_consistency(self):
        history=object.__new__(VerifiedChunkHistory)
        history.initial_time=.01;history._clock_candidates=np.ones(2,dtype=bool)
        steps=np.arange(1,36001)
        actual=.01+steps*float(np.float32(.005))
        self.assertGreater(abs(actual[-1]-(.01+steps[-1]*.005)),1e-6)
        for start in range(0,len(steps),500):history._check_actual_clock(steps[start:start+500],actual[start:start+500])
        self.assertEqual(history._clock_candidates.tolist(),[False,True])
        with self.assertRaises(GeometryNotVerified):history._check_actual_clock(steps[-500:],.01+steps[-500:]*.005)
        for invalid in (actual[-500:]+.005,.01+steps[-500:]*.005000001):
            fresh=object.__new__(VerifiedChunkHistory);fresh.initial_time=.01;fresh._clock_candidates=np.ones(2,dtype=bool)
            with self.assertRaises(GeometryNotVerified):fresh._check_actual_clock(steps[-500:],invalid)

    def test_complete_stream_two_chunks_and_initial(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            run=Path(directory);cap,geom,index=fixture(run)
            history=iterate_verified_chunks(run,cap,geom);iterator=iter(history)
            self.assertEqual(next(iterator)["time_s"].shape,(500,))
            self.assertEqual(history.summary["status"],"NOT_FULLY_CONSUMED")
            self.assertEqual(next(iterator)["time_s"].shape,(3,))
            self.assertEqual(history.summary["status"],"NOT_FULLY_CONSUMED")
            with self.assertRaises(StopIteration):next(iterator)
            self.assertEqual(history.summary["status"],"VERIFIED_HISTORY")
            self.assertEqual(history.summary["zero_step_phase_count"],1)
            self.assertEqual(len(history.summary["source_files"]),6)
            self.assertEqual(history.initial_arrays["body_link_pose_w"].shape,(1,7))

    def test_wrong_hash_initial_index_chunk(self):
        for kind in ("initial","index","chunk"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
                run=Path(directory);cap,geom,index=fixture(run)
                path=run/{"initial":"initial.npz","index":"index.json","chunk":"chunk_1.npz"}[kind]
                with path.open("ab") as out:out.write(b"corruption")
                with self.assertRaises(GeometryNotVerified):verify_history(run,cap,geom)

    def test_semantic_corruption_with_rehashed_bytes(self):
        mutations={
            "cross_chunk_missing_step":lambda a:a["physics_step"].__setitem__(0,502),
            "real_clock_gap":lambda a:a["actual_simulation_time_s"].__setitem__(0,a["actual_simulation_time_s"][0]+.005),
            "wrong_body_count":lambda a:a.__setitem__("body_positions_world",np.zeros((3,2,3))),
            "nonfinite_nodes":lambda a:a["nodes_world"].__setitem__((0,0,0),np.nan),
            "zero_phase_fake_sample":lambda a:a["phase_instance_id"].__setitem__(0,2),
            "wrong_phase_name":lambda a:a["phase"].__setitem__(0,"fake_phase"),
            "fake_early_boundary":lambda a:a["boundary"].__setitem__(1,True),
            "missing_final_boundary":lambda a:a["boundary"].__setitem__(-1,False),
            "nonunit_quaternion":lambda a:a["body_quaternions_wxyz"].__setitem__((0,0,0),2.),
        }
        for name,mutate in mutations.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
                run=Path(directory);cap,geom,index=fixture(run)
                change_chunk(run,index,1,mutate);refresh(run,cap,index)
                with self.assertRaises(GeometryNotVerified):verify_history(run,cap,geom)

    def test_ledger_and_absolute_anchor_corruption(self):
        changes=[lambda c:c["execution_anchors"].__setitem__("independently_expected_total_steps",504),
                 lambda c:c["phase_execution_ledger"][2].__setitem__("final_snapshot_physics_step",501),
                 lambda c:c["phase_execution_ledger"][3].__setitem__("call_status","EXCEPTION"),
                 lambda c:c["phase_execution_ledger"][3].__setitem__("phase","unrelated_final"),
                 lambda c:c["initial_state"].__setitem__("physics_step",1)]
        for change in changes:
            with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
                run=Path(directory);cap,geom,index=fixture(run);change(cap);refresh(run,cap,index)
                with self.assertRaises(GeometryNotVerified):verify_history(run,cap,geom)

    def test_missing_suffix_and_legacy_are_not_verified(self):
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as directory:
            run=Path(directory);cap,geom,index=fixture(run);index["chunks"].pop();refresh(run,cap,index)
            with self.assertRaises(GeometryNotVerified):verify_history(run,cap,geom)
            cap["history_layout"]="SINGLE_NPZ_V1";cap["nominal_hz"]=50;refresh(run,cap,index)
            with self.assertRaises(GeometryNotVerified):verify_history(run,cap,geom)


if __name__ == "__main__":unittest.main(verbosity=2)
