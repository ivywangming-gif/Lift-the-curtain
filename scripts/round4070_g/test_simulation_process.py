"""CPU-only process-guard regressions; no simulator or model is launched."""
import datetime as dt
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation_process import is_simulation_process, python_entrypoint


class SimulationProcessContracts(unittest.TestCase):
    def test_recorded_git_pathspecs_and_reader_operands_are_not_simulation(self):
        # Actual false-positive form captured by the host process monitor.
        paths = [":(literal)scripts/round4070_g/g1_trial.py",
                 ":(literal)scripts/round4070_failure_discovery/run_episode.py",
                 ":(literal)scripts/round4070_g/physics_gate.py"]
        cases = [
            ("git.exe", [r"C:\Program Files\Git\cmd\git.exe", "diff", "--", *paths]),
            ("rg.exe", ["rg.exe", "AppLauncher", "scripts/round4070_g/g1_trial.py"]),
            ("python.exe", ["python.exe", "-m", "py_compile", "scripts/round4070_g/g1_trial.py"]),
            ("python.exe", ["python.exe", "-m", "unittest", "g1_trial.py"]),
            ("python.exe", ["python.exe", "-c", "print('read only')", "g1_trial.py"]),
            ("python.exe", ["python.exe", "read.py", "g1_trial.py"]),
            ("python.exe", ["python.exe", "-X", "g1_trial.py", "read.py"]),
            ("python.exe", ["python.exe", "-W", "g1_trial.py"]),
            ("python.exe", ["python.exe", "-", "g1_trial.py"]),
            ("python.exe", ["python.exe", "--help", "g1_trial.py"]),
            ("cmd.exe", ["cmd.exe", "/c", "python.exe", "g1_trial.py"]),
            ("git.exe", ["python.exe", "g1_trial.py"]),  # actual executable is authoritative
        ]
        for name, command in cases:
            with self.subTest(name=name, command=command):
                self.assertFalse(is_simulation_process(name, command))

    def test_python_options_modules_and_native_simulators_are_recognized(self):
        entry = r"C:\Robotics\Lift-the-curtain\scripts\round4070_g\g1_trial.py"
        variants = [
            ["python.exe", entry, "--headless"],
            ["python.exe", "-u", "-X", "utf8", entry, "--headless"],
            ["python.exe", "-Bu", "-Xutf8", "-Wignore", entry],
            ["python.exe", "-uX", "utf8", "-W", "ignore", entry],
            ["python.exe", "--check-hash-based-pycs", "always", "--", entry],
            ["python.exe", "-m", "scripts.round4070_g.g1_trial", "--headless"],
            ["python.exe", "-u", "physics_gate.py"],
            ["python.exe", "physics_collision_smoke.py"],
            ["python.exe", "physics_wall_smoke.py"],
            ["python.exe", "physics_f05_initialization_diagnostic.py"],
        ]
        for command in variants:
            with self.subTest(command=command):
                self.assertTrue(is_simulation_process("python.exe", command))
        self.assertTrue(is_simulation_process("PYTHONW.EXE", ["pythonw.exe", entry]))
        self.assertTrue(is_simulation_process("python3.11", ["python3.11", entry]))
        self.assertTrue(is_simulation_process("kit.exe", []))
        self.assertTrue(is_simulation_process("isaac-sim.exe", ["isaac-sim.exe", "--no-window"]))
        self.assertEqual(python_entrypoint(["python", "-X", "utf8", "read.py", entry]), ("script", "read.py"))

    def test_workflow_and_trainer_share_matcher_without_weakening_window(self):
        import workflow
        import train_benchmark
        self.assertIs(workflow.is_simulation_process, train_benchmark.is_simulation_process)
        git = SimpleNamespace(info=dict(pid=9876541, name="git.exe", cmdline=["git.exe", "diff", ":(literal)scripts/round4070_g/g1_trial.py"]))
        compile_only = SimpleNamespace(info=dict(pid=9876542, name="python.exe", cmdline=["python.exe", "-m", "py_compile", "g1_trial.py"]))
        actual = SimpleNamespace(info=dict(pid=9876543, name="python.exe", cmdline=["python.exe", "-u", "-X", "utf8", "g1_trial.py"]))
        parent = workflow.ROOT/"outputs/round4070_comprehensive_traversability/contract_tests"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            window = Path(directory)/"window.json"
            payload = dict(allow_training=True, isaac_stopped_confirmed=True,
                           expires_utc=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(minutes=1)).isoformat())
            window.write_text(json.dumps(payload))
            with patch("psutil.process_iter", return_value=[git, compile_only]):
                self.assertEqual(workflow.active_simulations(), [])
                self.assertEqual(train_benchmark.window_status(window), (True, "AUTHORIZED_WINDOW"))
            with patch("psutil.process_iter", return_value=[git, compile_only, actual]):
                self.assertEqual(workflow.active_simulations(), [9876543])
                self.assertEqual(train_benchmark.window_status(window), (False, "SIMULATION_PROCESS_ACTIVE:9876543"))
            for change, reason in [
                ({"allow_training":False}, "SIMULATION_FREE_WINDOW_NOT_CONFIRMED"),
                ({"isaac_stopped_confirmed":False}, "SIMULATION_FREE_WINDOW_NOT_CONFIRMED"),
                ({"expires_utc":"2000-01-01T00:00:00+00:00"}, "TRAINING_WINDOW_EXPIRED"),
            ]:
                window.write_text(json.dumps({**payload, **change}))
                self.assertEqual(train_benchmark.window_status(window), (False, reason))
            self.assertEqual(train_benchmark.window_status(Path(directory)/"missing.json"), (False, "MISSING_EXPLICIT_TRAINING_WINDOW"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
