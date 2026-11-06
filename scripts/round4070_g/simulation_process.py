"""Identify real simulator entrypoints, not filenames in unrelated arguments.

This is a supplementary process guard. Resource locks and an explicit expiring
training window remain mandatory. Opaque Python -c/stdin programs and shell
wrappers are not guessed from arbitrary argument text; the actual simulator
Python/Kit child is what this matcher recognizes.
"""
from __future__ import annotations

import ntpath
import re


SIMULATION_SCRIPTS = frozenset({
    "run_episode.py", "physics_p1_asset.py", "run_physics_scene.py",
    "physics_scene.py", "physics_gate.py", "g1_trial.py",
    "physics_collision_smoke.py", "physics_wall_smoke.py",
    "physics_f05_initialization_diagnostic.py",
})
NATIVE_EXECUTABLES = frozenset({"kit.exe", "isaac-sim.exe", "kit", "isaac-sim"})
PYTHON_EXECUTABLE = re.compile(r"pythonw?(?:\d+(?:\.\d+)*)?(?:\.exe)?\Z")


def _basename(value):
    # ntpath handles both Windows separators even in simulator-free tests.
    return ntpath.basename(str(value).strip('"')).casefold()


def python_entrypoint(command):
    """Return (script/module, value) for CPython's actual execution target.

    Parse interpreter options only, stopping at the first execution target.
    In particular, -X/-W consume their values, and -m py_compile cannot turn
    a subsequent g1_trial.py operand into the executed simulation entrypoint.
    """
    index = 1
    while index < len(command):
        token = str(command[index])
        if token == "--":
            index += 1
            return ("script", str(command[index])) if index < len(command) and command[index] != "-" else None
        if token == "-":
            return None
        if not token.startswith("-"):
            return "script", token
        if token in ("--help", "--help-env", "--help-xoptions", "--help-all", "--version"):
            return None
        if token == "--check-hash-based-pycs":
            index += 2
            continue
        if token.startswith("--check-hash-based-pycs="):
            index += 1
            continue
        if token.startswith("--"):
            return None  # Unknown interpreter option is not a script operand.
        options = token[1:]
        while options:
            option, options = options[0], options[1:]
            if option in "bBdEiIOPqRsSuvx":
                continue
            if option in "h?V":
                return None
            if option in "cmWX":
                if options:
                    value = options
                else:
                    index += 1
                    if index >= len(command):
                        return None
                    value = str(command[index])
                if option == "c":
                    return None
                if option == "m":
                    return "module", value
                break  # -W/-X value consumed; continue with the next argv.
            return None
        index += 1
    return None


def is_simulation_process(name, command):
    """Match psutil's actual executable and Python entrypoint only."""
    command = command or []
    executable = _basename(name or (command[0] if command else ""))
    if executable in NATIVE_EXECUTABLES:
        return True
    if not PYTHON_EXECUTABLE.fullmatch(executable):
        return False
    entry = python_entrypoint(command)
    if entry is None:
        return False
    kind, target = entry
    if kind == "module":
        return target.rsplit(".", 1)[-1].casefold()+".py" in SIMULATION_SCRIPTS
    return _basename(target) in SIMULATION_SCRIPTS
