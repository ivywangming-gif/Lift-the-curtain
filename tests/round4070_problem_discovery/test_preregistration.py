import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_initial_trial_matrix_is_frozen_and_within_budget() -> None:
    data = json.loads((REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    trials = data["trials"]
    assert len(trials) == data["initial_matrix_count"] == 22
    assert len({trial["trial_id"] for trial in trials}) == len(trials)
    assert len(trials) + data["confirmatory_budget_reserved"] + data["active_probe_budget_reserved"] <= data["new_g1_trial_hard_limit"]
    assert all(trial["policy"] == "r1" for trial in trials)
    assert {trial["probe_side"] for trial in trials} == {"left", "right"}


def test_early_match_rule_is_declared_before_runs() -> None:
    data = json.loads((REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    rule = data["frozen_early_match_rule"]
    assert rule["cutoff"].startswith("T1")
    assert rule["maximum_absolute_differences"]["hand_forward_delta_m"] > 0.0


def test_confirmatory_matrix_is_independent_and_within_budget() -> None:
    initial = json.loads((REPO / "configs" / "round4070_problem_discovery" / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    confirmation = json.loads((REPO / "configs" / "round4070_problem_discovery" / "CONFIRMATORY_PREREGISTRATION.json").read_text(encoding="utf-8"))
    initial_ids = {trial["trial_id"] for trial in initial["trials"]}
    trials = confirmation["trials"]
    assert len(trials) == confirmation["trial_count"] == 16
    assert not initial_ids.intersection(trial["trial_id"] for trial in trials)
    assert len(initial["trials"]) + len(trials) + confirmation["hard_budget_accounting"]["launches_reserved_after_confirmation"] == initial["new_g1_trial_hard_limit"]
    assert {trial["seed"] for trial in trials} == {49, 50}
    assert all(trial["stage"] == "confirmatory" for trial in trials)
