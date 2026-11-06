"""Generate G learning tables only from completed, traceable fit artifacts."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics

import numpy as np

from learning_core import ROOT, read_json, sha, write_csv, write_json


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def average_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split_id"], row["method"], row["mode"]].append(row)
    out = []
    for (split_id, method, mode), values in sorted(grouped.items()):
        numeric = ["false_pass_n", "false_reject_n", "true_pass_n", "true_block_n", "false_pass", "false_reject", "passable_recall", "blocked_recall", "balanced_accuracy", "coverage", "uncertain_rate", "selective_risk", "auroc", "auprc", "brier", "ece", "parameters", "training_seconds", "inference_ms"]
        row = dict(split_id=split_id, method=method, mode=mode, actual_fits=len(values), actual_seeds="|".join(str(v["seed"]) for v in values),
                   evaluation_scope=values[0].get("evaluation_scope", "FULL_SPLIT_BENCHMARK"),
                   base_method=values[0].get("base_method", method), label_fraction=values[0].get("label_fraction", 1.),
                   supervised_train_n=values[0].get("supervised_train_n"),
                   n=values[0]["n"], passable_n=values[0]["passable_n"], blocked_n=values[0]["blocked_n"],
                   repeat_interpretation="same physical test episodes; seed SD is optimization variability only")
        for key in numeric:
            v = [float(x[key]) for x in values if x.get(key) not in (None, "", "None")]
            row[key+"_mean"] = statistics.mean(v) if v else None
            row[key+"_std"] = statistics.stdev(v) if len(v) > 1 else None
        out.append(row)
    return out


def paired_group_bootstrap(predictions, seeds, reference="GRU"):
    """Fixed-policy paired differences: average seeds first, bootstrap groups."""
    by_split = defaultdict(lambda: defaultdict(dict))
    for r in predictions:
        mode = r["mode"].removeprefix("confirmation/")
        endpoint = "confirmation" if r["mode"].startswith("confirmation/") else "main_test"
        if (mode == "forced_raw" or mode.startswith("calibration_risk_")) and r["y"] in ("0", "1") and r.get("is_control") not in ("True", "true", "1"):
            by_split[endpoint, r["split_id"], float(r.get("label_fraction", 1.)), mode][r["method"]][r["trial_id"], r["seed"]] = r
    output = []
    definitions = {"false_pass": (0, lambda d: d == 1), "false_reject": (1, lambda d: d == 0),
                   "coverage": (None, lambda d: d >= 0), "passable_recall": (1, lambda d: d == 1)}
    for (endpoint, split, fraction, mode), methods in by_split.items():
        local_reference = reference if fraction == 1. else f"{reference}__labels{int(fraction*100):03d}"
        if local_reference not in methods:
            continue
        ref = methods[local_reference]
        for method, records in methods.items():
            if method == local_reference:
                continue
            candidates = {trial for trial, seed in ref} & {trial for trial, seed in records}
            paired_episodes = []
            for trial in sorted(candidates):
                keys = [(trial, str(s)) for s in seeds]
                if not all(k in ref and k in records for k in keys):
                    continue
                truth = int(ref[keys[0]]["y"])
                group = ref[keys[0]]["group_id"]
                if any(int(records[k]["y"]) != truth or int(ref[k]["y"]) != truth or records[k]["group_id"] != group or ref[k]["group_id"] != group for k in keys):
                    raise ValueError("Paired predictions disagree on physical episode truth/group")
                paired_episodes.append((truth, group, [int(records[k]["decision"]) for k in keys], [int(ref[k]["decision"]) for k in keys]))
            for metric, (target_class, contribution) in definitions.items():
                differences = [(group, float(np.mean([int(contribution(a))-int(contribution(b)) for a, b in zip(current, baseline)])))
                               for truth, group, current, baseline in paired_episodes if target_class is None or truth == target_class]
                groups = sorted({g for g, value in differences})
                difference = float(np.mean([v for g, v in differences])) if differences else None
                row = dict(evaluation_endpoint=endpoint, split_id=split, mode=mode, metric=metric,
                           method=method, reference=local_reference, label_fraction=fraction,
                           paired_episode_n=len(differences), physical_group_n=len(groups), paired_seed_n=len(seeds),
                           mean_difference=difference, negative_episode_n=len(differences) if metric == "false_pass" else None,
                           mean_false_pass_rate_difference=difference if metric == "false_pass" else None,
                           ci95_low=None, ci95_high=None,
                           interpretation="descriptive, multiplicity-unadjusted configuration-cluster CI; seed-averaged paired episode contributions; selected configurations; not a significance or deployment guarantee")
                if mode.startswith("calibration_risk_"):
                    row["interpretation"] += "; same calibration target, not guaranteed equal heldout risk; assess FPR and coverage jointly"
                if len(groups) >= 3:
                    chunks = {g: np.asarray([v for group, v in differences if group == g]) for g in groups}
                    rng = np.random.default_rng(407099)
                    draws = rng.integers(0, len(groups), size=(1000, len(groups)))
                    group_sums = np.asarray([chunks[g].sum() for g in groups])
                    group_counts = np.asarray([len(chunks[g]) for g in groups])
                    samples = group_sums[draws].sum(1)/group_counts[draws].sum(1)
                    row["ci95_low"], row["ci95_high"] = np.quantile(samples, [.025, .975]).tolist()
                elif differences:
                    row["interpretation"] += "; CI unavailable: fewer than three eligible physical groups"
                else:
                    row["interpretation"] += "; UNAVAILABLE: no complete paired labelled episodes for this metric"
                output.append(row)
    return output


def generate(run, report):
    protocol = read_json(run/"frozen_protocol.json")
    queue = read_json(run/"task_queue.json")
    report.mkdir(parents=True, exist_ok=True)
    from learning_models import METHOD_GROUPS, ENGINEERING
    metrics_rows, predictions, bins, evidence, confirmation_states = [], [], [], [], []
    split_map = {s["split_id"]: s for s in protocol["split_definitions"]}
    for task in queue["tasks"]:
        folder = run/"fits"/task["task_id"]
        split = split_map[task["split_id"]]
        confirmation_state = read_json(folder/"confirmation_status.json") if task["status"] == "EVALUATED" and (folder/"confirmation_status.json").exists() else dict(
            status="NOT_EVALUATED", task_status=task["status"], task_reasons="|".join(task["reasons"]),
            declared_confirmation_status=split.get("confirmation_status", "NOT_DECLARED_IN_OLD_PROTOCOL"),
            confirmation_ids=split.get("confirmation", []), core_labelled_n=None)
        confirmation_states.append(dict(task_id=task["task_id"], split_id=task["split_id"], method=task.get("display_method", task["method"]),
            evaluation_scope=task.get("evaluation_scope", "FULL_SPLIT_BENCHMARK"),
            seed=task["seed"], label_fraction=task.get("label_fraction", 1.), **confirmation_state))
        if task["status"] != "EVALUATED":
            continue
        complete = read_json(folder/"completion.json")
        if sha(folder/"predictions.csv") != complete["predictions_sha256"]:
            raise ValueError("Completed prediction artifact changed")
        metrics_rows.extend(read_csv(folder/"metrics.csv"))
        predictions.extend(read_csv(folder/"predictions.csv"))
        if (folder/"calibration_bins.csv").exists():
            bins.extend(read_csv(folder/"calibration_bins.csv"))
        evidence.append(dict(task_id=task["task_id"], completion=str(folder/"completion.json"), metrics=str(folder/"metrics.csv"),
                             metrics_sha256=sha(folder/"metrics.csv"), predictions_sha256=complete["predictions_sha256"], checkpoint_sha256=complete.get("checkpoint_sha256")))
    partial_rows = [r for r in metrics_rows if r.get("evaluation_scope") == "PARTIAL_HELDOUT_SINGLE_CLASS"]
    partial_predictions = [r for r in predictions if r.get("evaluation_scope") == "PARTIAL_HELDOUT_SINGLE_CLASS"]
    partial_bins = [r for r in bins if r.get("evaluation_scope") == "PARTIAL_HELDOUT_SINGLE_CLASS"]
    partial_schema = ["evaluation_scope", "split_id", "method", "seed", "mode", "n", "passable_n", "blocked_n", "false_pass_n", "false_reject_n", "coverage", "balanced_accuracy", "auroc", "selective_risk"]
    write_csv(report/"PARTIAL_HELDOUT_SINGLE_CLASS_RESULTS.csv", partial_rows, None if partial_rows else partial_schema)
    write_csv(report/"PARTIAL_HELDOUT_SINGLE_CLASS_PREDICTIONS.csv", partial_predictions, None if partial_predictions else ["evaluation_scope", "split_id", "method", "seed", "mode", "trial_id", "y", "p", "decision"])
    partial_mean = average_rows([r for r in partial_rows if r["subset"] == "all"])
    write_csv(report/"PARTIAL_HELDOUT_SINGLE_CLASS_MEAN_STD.csv", partial_mean, None if partial_mean else ["evaluation_scope", "split_id", "method", "mode", "actual_fits"])
    write_csv(report/"PARTIAL_HELDOUT_SINGLE_CLASS_CALIBRATION_BINS.csv", partial_bins, None if partial_bins else ["evaluation_scope", "split_id", "method", "probability_kind", "bin", "n"])
    metrics_rows = [r for r in metrics_rows if r.get("evaluation_scope") != "PARTIAL_HELDOUT_SINGLE_CLASS"]
    predictions = [r for r in predictions if r.get("evaluation_scope") != "PARTIAL_HELDOUT_SINGLE_CLASS"]
    bins = [r for r in bins if r.get("evaluation_scope") != "PARTIAL_HELDOUT_SINGLE_CLASS"]
    all_predictions = predictions
    confirmation_rows = [r for r in metrics_rows if r["mode"].startswith("confirmation/")]
    confirmation_predictions = [r for r in predictions if r["mode"].startswith("confirmation/")]
    confirmation_bins = [r for r in bins if r["probability_kind"].startswith("confirmation/")]
    metrics_rows = [r for r in metrics_rows if not r["mode"].startswith("confirmation/")]
    predictions = [r for r in predictions if not r["mode"].startswith("confirmation/")]
    bins = [r for r in bins if not r["probability_kind"].startswith("confirmation/")]
    all_rows = [r for r in metrics_rows if r["subset"] == "all"]
    schema = ["split_id", "method", "base_method", "label_fraction", "supervised_train_n", "seed", "mode", "n", "passable_n", "blocked_n", "true_pass_n", "true_block_n", "false_pass_n", "false_reject_n", "uncertain_n", "coverage"]
    confirmation_all = [r for r in confirmation_rows if r["subset"] == "all"]
    write_csv(report/"CONFIRMATION_RESULTS.csv", confirmation_all, None if confirmation_all else schema)
    write_csv(report/"CONFIRMATION_ALL_SUBGROUP_METRICS.csv", confirmation_rows, None if confirmation_rows else schema)
    write_csv(report/"CONFIRMATION_PREDICTIONS.csv", confirmation_predictions, None if confirmation_predictions else ["split_id", "method", "seed", "mode", "trial_id", "y", "p", "decision"])
    write_csv(report/"CONFIRMATION_MEAN_STD.csv", average_rows(confirmation_all), None if confirmation_all else ["split_id", "method", "mode", "actual_fits"])
    write_csv(report/"CONFIRMATION_STATUS.csv", confirmation_states, None if confirmation_states else ["task_id", "status", "confirmation_ids"])
    write_csv(report/"CONFIRMATION_CALIBRATION_BINS.csv", confirmation_bins, None if confirmation_bins else ["split_id", "method", "seed", "probability_kind", "bin", "n"])
    # Empty files contain schema only. The status manifest explicitly says no
    # runs; zero rows never masquerade as zero errors or completed comparisons.
    names = {
        "ENGINEERING_BASELINE_RESULTS.csv": set(ENGINEERING),
        "STANDARD_LEARNING_BASELINES.csv": set(METHOD_GROUPS["standard"]),
        "CONTRASTIVE_LEARNING_RESULTS.csv": set(METHOD_GROUPS["contrastive"]),
        "TEMPORAL_EVIDENCE_RESULTS.csv": set(METHOD_GROUPS["temporal"]),
        "RESPONSE_PRETRAINING_RESULTS.csv": set(METHOD_GROUPS["response"]),
        "OBSERVATION_DR_TRAINING_RESULTS.csv": set(METHOD_GROUPS["observation_dr"]),
    }
    for filename, methods in names.items():
        selected = [r for r in all_rows if r["method"] in methods and float(r.get("label_fraction", 1.)) == 1.]
        write_csv(report/filename, selected, None if selected else schema)
    efficiency = [r for r in all_rows if r.get("base_method", r["method"]) in ("GRU", "GRUResponseAux", "GRUPretrainFinetune")]
    write_csv(report/"LABEL_EFFICIENCY_RESULTS.csv", efficiency, None if efficiency else schema)
    budget_rows = []
    for split_id, budgets in protocol.get("label_efficiency", {}).get("subsets", {}).items():
        for fraction, budget in budgets.items():
            budget_rows.append(dict(split_id=split_id, label_fraction=fraction,
                original_labelled_train_n=budget["original_labelled_train_n"], selected_labelled_train_n=budget["selected_labelled_train_n"],
                selected_train_label_ids="|".join(budget["selected_train_label_ids"]), hidden_train_label_ids="|".join(budget["hidden_train_label_ids"]),
                selected_passable_n=budget["selected_class_counts"]["1"], selected_blocked_n=budget["selected_class_counts"]["0"],
                selected_core_passable_n=budget["selected_core_class_counts"]["1"], selected_core_blocked_n=budget["selected_core_class_counts"]["0"],
                scope=budget["scope"]))
    write_csv(report/"LABEL_BUDGET_FROZEN_SUBSETS.csv", budget_rows, None if budget_rows else ["split_id", "label_fraction", "selected_labelled_train_n", "selected_train_label_ids", "hidden_train_label_ids"])
    kinds = {s["split_id"]: s.get("split_type", s.get("kind", "configuration")) for s in protocol["split_definitions"]}
    cross = [r for r in all_rows if kinds[r["split_id"]] in ("cross_hand", "posture", "legacy_side") or r["method"] in METHOD_GROUPS["cross_hand"]]
    ood = [r for r in all_rows if kinds[r["split_id"]] != "iid_seed"]
    write_csv(report/"CROSS_HAND_GENERALIZATION.csv", cross, None if cross else schema)
    write_csv(report/"OOD_GENERALIZATION_RESULTS.csv", ood, None if ood else schema)
    write_csv(report/"LEARNING_ALL_SUBGROUP_METRICS.csv", metrics_rows, None if metrics_rows else schema)
    sensor_rows = [r for r in all_rows if r["mode"].startswith("noisy_test/") or r["method"] in METHOD_GROUPS["observation_dr"]]
    write_csv(report/"OBSERVATION_RANDOMIZATION_RESULTS.csv", sensor_rows, None if sensor_rows else schema)
    write_csv(report/"LEARNING_PREDICTIONS.csv", predictions, None if predictions else ["split_id", "method", "seed", "mode", "trial_id", "group_id", "side", "y", "p", "decision"])
    summary = average_rows(all_rows)
    write_csv(report/"LEARNING_MEAN_STD.csv", summary, None if summary else ["split_id", "method", "mode", "actual_fits"])
    write_csv(report/"CALIBRATION_BINS.csv", bins, None if bins else ["split_id", "method", "seed", "probability_kind", "bin", "n", "predicted_mean", "observed_passable_rate"])
    pairs = paired_group_bootstrap(all_predictions, protocol["seeds"])
    pair_schema = ["evaluation_endpoint", "split_id", "method", "reference", "mode", "metric", "label_fraction", "paired_episode_n", "physical_group_n", "paired_seed_n", "mean_difference", "ci95_low", "ci95_high"]
    for filename, endpoint in (("PAIRED_CONFIGURATION_ANALYSIS.csv", "main_test"), ("CONFIRMATION_PAIRED_CONFIGURATION_ANALYSIS.csv", "confirmation")):
        selected = [r for r in pairs if r["evaluation_endpoint"] == endpoint]
        write_csv(report/filename, selected, None if selected else pair_schema)
    write_csv(report/"LEARNING_RUN_STATUS.csv", [{**t, "reasons": "|".join(t["reasons"])} for t in queue["tasks"]],
              None if queue["tasks"] else ["task_id", "split_id", "method", "seed", "status", "reasons"])
    write_csv(report/"LEARNING_ARTIFACT_INDEX.csv", evidence, None if evidence else ["task_id", "completion", "metrics", "metrics_sha256", "predictions_sha256", "checkpoint_sha256"])
    full_tasks = [t for t in queue["tasks"] if t.get("evaluation_scope") != "PARTIAL_HELDOUT_SINGLE_CLASS"]
    partial_tasks = [t for t in queue["tasks"] if t.get("evaluation_scope") == "PARTIAL_HELDOUT_SINGLE_CLASS"]
    count_status = lambda tasks: {status:sum(t["status"]==status for t in tasks) for status in sorted({t["status"] for t in tasks})}
    counts = count_status(full_tasks)
    state = dict(scope=protocol["scope"], evaluated_fit_n=sum(t["status"]=="EVALUATED" for t in full_tasks), planned_task_n=len(full_tasks), task_counts=counts,
                 all_scopes_evaluated_task_n=len(evidence),all_scopes_planned_task_n=len(queue["tasks"]),all_scopes_task_counts=count_status(queue["tasks"]),
                 partial_single_class_planned_task_n=len(partial_tasks),partial_single_class_task_counts=count_status(partial_tasks),
                 full_benchmark_evaluated_fit_n=sum(t["status"] == "EVALUATED" and t.get("evaluation_scope") != "PARTIAL_HELDOUT_SINGLE_CLASS" for t in queue["tasks"]),
                 partial_single_class_evaluated_fit_n=sum(t["status"] == "EVALUATED" and t.get("evaluation_scope") == "PARTIAL_HELDOUT_SINGLE_CLASS" for t in queue["tasks"]),
                 partial_single_class_scope="separate descriptive diagnostic; does not complete any original full split",
                 run=str(run), protocol_sha256=sha(run/"frozen_protocol.json"),
                 benchmark_completion="NOT_INFERRED_FROM_MODEL_EXECUTION: physical campaign and coverage gates remain separate",
                 empty_table_meaning="NO_COMPLETED_ACTUAL_RUNS; never zero errors", legacy_only=protocol["legacy"])
    write_json(report/"LEARNING_DELIVERY_STATUS.json", state)
    risk = [r for r in all_rows if r["mode"].startswith("calibration_risk_")]
    write_csv(report/"RISK_COVERAGE_POINTS.csv", risk, None if risk else schema)
    report.joinpath("RISK_COVERAGE_ANALYSIS.md").write_text(
        "# G 风险、覆盖率与校准\n\n"
        + f"范围：`{protocol['scope']}`。真实已评价 fit：{len(evidence)} / 预定任务 {len(queue['tasks'])}。状态 `{counts}`。\n\n"
        + "forced=0.5、固定拒判=.2/.8；温度拟合和风险阈值只用独立 calibration 组。阈值采用预先固定的 .025 网格，目标误放行率为 0/.01/.025/.05/.1/.2，同时约束校准 false-reject≤.1，再最大化确定覆盖率。该约束是校准样本上的经验约束，不是部署安全认证。缺任一校准类别则不拟合温度/经验阈值。\n\n"
        + "label-conditional split conformal 的 alpha=.1，分数使用冻结模型的raw概率，不能复用同calibration标签拟合temperature后的分数。有限样本分位数超出可用秩时使用 q=1 的保守集合，双类/空集合都输出 UNCERTAIN。物理配置采集存在选择偏差，交换性未验证，不宣称真实世界覆盖保证。\n\n"
        + "FP分母为全部可靠BLOCKED，FR为全部可靠PASSABLE；拒判不算TN/TP。selective risk只在已决策集合计错误，必须同时报告coverage与PASSABLE recall。UNKNOWN永远不进监督分母。工程规则的0/.5/1是决定编码，不作为概率报告Brier或校准。\n\n"
        + "`RISK_COVERAGE_POINTS.csv`、`CALIBRATION_BINS.csv`、各fit的calibration.json均来自实际执行。有实际有效bin时生成 `CALIBRATION_RELIABILITY.png`，按固定候选方法展示 clean test 的 raw/temperature 可靠性曲线；汇总权重是预测数，不把重复seed当独立物理样本、不画未经支持的误差带。完整方法及noisy诊断都保留在CSV。只有表头表示尚无实际可用评价，不是零风险。\n\n"
        + "`PAIRED_CONFIGURATION_ANALYSIS.csv`在forced和各预注册calibration风险模式下分别评价FPR/FR/coverage/PASSABLE recall差：先同episode平均五seed，再抽物理配置组；不足3个相应类别/总体配置不报告CI。CI为描述性、未多重比较校正，不能据此宣称显著新方法。相同calibration风险目标不保证相同heldout风险，coverage差必须与FPR联合解释。`false_pass_episode_binomial_wilson_*`仅为假设episode IID的二项描述区间，不是物理配置cluster区间或跨配置保证。\n\n"
        + "confirmation是第四个独立角色，完全排除train/calibration/main test。每个预注册模型和预算用同一已冻结模型及main calibration参数独立评价；不重新拟温度/阈值、不选赢家。`CONFIRMATION_*`表与主表分开，状态列明缺样本/缺类别/未运行。确认集只做clean诊断，未声称已做其观测DR。\n", encoding="utf-8")
    report.joinpath("ABLATION_REPORT.md").write_text(
        "# G 消融执行审计\n\n"
        + f"真实已评价fit：{len(evidence)}；详细未运行/受阻原因见 `LEARNING_RUN_STATUS.csv`。当前范围 `{protocol['scope']}`，不据模型数量宣布完整benchmark完成。\n\n"
        + "M2候选为标准监督对比或有证据的物理配对cosine contrastive，非原创对比算法。完整匹配及两个消融要求同一个训练集同时具有真实困难负对照和跨配置正对照；缺失则全部明确受阻，不复制/镜像标签。离线元数据不进入特征。\n\n"
        + "M3包括真实观测历史、固定seed乱序、删累计手部进展的末观测、命令进程、真实dt+进程、固定.5/1s前缀及按结构参数数匹配的GRU。只保留真实观测；高密度数据最多128点，用固定时间网格的past索引去重，低频日志不插值造高频。参数匹配不使用标签/测试。\n\n"
        + "同episode的GRU10Hz/GRU20Hz使用相同2s截止及真实past观测去重，保留安全早停；基准实际频率以源数据审计为准，不以模型名称假定50Hz。所有方法都可从已获得样本长度推知提前停止，属于合法但可能依赖安全策略的线索；DurationOnlyLogistic仅用已获得点数/观测时长作该捷径诊断。GRUShuffled与普通GRU同预算通道，均删除显式elapsed/dt/command_progress；已执行q_cmd本身仍可携带阶段线索，乱序不能单独证明无时序信息。\n\n"
        + "M4联合响应辅助使用训练组（含UNKNOWN）的下一已记录q/qdot；分阶段版本40轮响应预训练+80轮监督微调，普通/联合为80轮，因此额外训练预算必须披露，不作纯损失因果结论。任何测试/校准UNKNOWN不进入拟合或归一化。\n\n"
        + "标签效率另设25/50/100%训练标签预算，三个方法GRU/GRUResponseAux/GRUPretrainFinetune及五seed共享同一固定hash、逐类floor的嵌套训练标签子集；100%复用原任务。预算计入已知训练controls，并单列核心P/B数；核心双类不足明确BLOCKED。未选标签只在训练视图改为UNKNOWN供响应学习，原manifest和校准/测试保持不变。`LABEL_BUDGET_FROZEN_SUBSETS.csv`记录实际IDs/数量，`LABEL_EFFICIENCY_RESULTS.csv`只记录实际完成结果；display method与label_fraction隔离统计，不把25%/50%和100%平均混为多seed。没有结果不能回答提高标签效率；额外预训练计算仍是混杂。\n\n"
        + "M5按真实关节名称重排active/opposite手并保留侧别，使用既有body-local FK；它不是经过运动学认证的带符号几何镜像。没有镜像标签或强制左右动作同标签。split7包含姿态/动作条件（包括速度），按robot_variant分层，不能把pool结果归因纯姿态。OldM2去FiLM的残差参照仍通过正常响应预测器间接使用身体状态，不是完全身体信息剥离。\n\n"
        + "观测DR有真实执行入口：GRUObsDR与普通GRU同结构/同clean训练scaler，只对训练原始观察作固定seed扰动；所有模型另用同一固定seed的noisy heldout诊断，校准概率始终保持clean。q=.001rad、qdot=.01rad/s、effort=训练尺度2%噪声/1%每条偏置、IMU角速度=.005rad/s/姿态=.001rad、时钟±.002s、丢样1%；延迟按真实50Hz可实现的0/20ms端点逐episode抽取，严格不插值。真实source年龄不得超过20ms；稀疏日志无法实现时单列不可用。尺度仅训练数据，元数据/标签不变；它是未硬件标定的仿真敏感性，并未耦合重算FK。source/receipt索引、seed、每条偏置与时钟保留audit；实际运行前结果表为空，不以实现代码宣称DR实验已完成。\n\n"
        + "需要从实际CSV解释优势/无优势；本生成器不会以尚未运行的代码或空表给方法有效性结论。\n", encoding="utf-8")
    report.joinpath("PARTIAL_HELDOUT_SINGLE_CLASS_SCOPE.md").write_text(
        "# 单类留出诊断范围\n\n"
        "冻结规则仅依据真实标签支持、从不依据预测结果：测试核心样本恰好一个类别时，单列100%标签预算、五个seed与同一预注册模型/工程基准。训练核心仍须双类，各方法原有可执行门槛保持；使用相同固定末epoch，不用测试或校准选checkpoint。原8个完整split状态保持不可评价。\n\n"
        "校准缺双类时，temperature/risk/conformal明确不可用，仍可报告raw forced-0.5与预设拒判策略的该类计数、错误率和覆盖率。BA/AUROC/双类selective risk为NA；class_conditional_selective_error仅描述当前支持类别。单类B不能回答P召回，单类P不能回答false pass；五seed只是优化重复，不是五倍独立物理样本。\n\n"
        "结果只写PARTIAL_HELDOUT_SINGLE_CLASS文件，不并入主学习/OOD/跨手/标签效率/校准结果，不用于宣称完整OOD或跨手泛化优势；confirmation仍独立且不参与训练/选择。空表表示没有实际完成的该诊断。\n",encoding="utf-8")
    if risk:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = defaultdict(list)
        for row in risk:
            grouped[row["split_id"], row["method"]].append(row)
        for (split, method), rows in grouped.items():
            if method not in ("GRU", "ResidualGRU", "GRUMatched", "GRUTimeProgress", "GRUResponseAux"):
                continue
            modes = sorted({r["mode"] for r in rows})
            pts = [(np.mean([float(r["coverage"]) for r in rows if r["mode"] == m]), np.mean([float(r["false_pass"]) for r in rows if r["mode"] == m])) for m in modes]
            ax.plot([v[0] for v in pts], [v[1] for v in pts], "o-", label=f"{split}: {method}")
        ax.set(xlabel="Actual test decision coverage", ylabel="Actual test false-pass rate", title="Calibration-chosen thresholds; no safety guarantee")
        if ax.lines:
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(report/"RISK_COVERAGE.png", dpi=160)
        plt.close(fig)
    valid_bins = [r for r in bins if r["probability_kind"] in ("raw", "temperature") and int(r["n"]) > 0]
    if valid_bins:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        candidates = {"Logistic", "GRU", "ResidualGRU", "GRUMatched", "GRUTimeProgress", "GRUObsDR"}
        for kind, ax in zip(("raw", "temperature"), axes):
            grouped = defaultdict(list)
            for row in valid_bins:
                if row["probability_kind"] == kind and row["method"] in candidates:
                    grouped[row["split_id"], row["method"], int(row["bin"])].append(row)
            curves = defaultdict(list)
            for (split, method, bin_id), rows in grouped.items():
                weights = np.asarray([int(r["n"]) for r in rows])
                predicted = np.average([float(r["predicted_mean"]) for r in rows], weights=weights)
                observed = np.average([float(r["observed_passable_rate"]) for r in rows], weights=weights)
                curves[split, method].append((bin_id, predicted, observed))
            for (split, method), points in sorted(curves.items()):
                points.sort()
                ax.plot([v[1] for v in points], [v[2] for v in points], "o-", label=f"{split}: {method}")
            ax.plot([0, 1], [0, 1], "k--", linewidth=.8)
            ax.set(xlabel="Mean predicted PASSABLE probability", ylabel="Observed PASSABLE fraction",
                   title=f"Clean heldout: {kind}", xlim=(0, 1), ylim=(0, 1))
            if curves:
                ax.legend(fontsize=5)
        fig.suptitle("Prediction-weighted descriptive bins; repeated seeds are not independent episodes", fontsize=9)
        fig.tight_layout()
        fig.savefig(report/"CALIBRATION_RELIABILITY.png", dpi=160)
        plt.close(fig)
    print(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, default=ROOT/"research/round4070_comprehensive_traversability")
    args = parser.parse_args()
    generate(args.run, args.report_dir)
