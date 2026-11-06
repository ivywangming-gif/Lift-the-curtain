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
                   n=values[0]["n"], passable_n=values[0]["passable_n"], blocked_n=values[0]["blocked_n"],
                   repeat_interpretation="same physical test episodes; seed SD is optimization variability only")
        for key in numeric:
            v = [float(x[key]) for x in values if x.get(key) not in (None, "", "None")]
            row[key+"_mean"] = statistics.mean(v) if v else None
            row[key+"_std"] = statistics.stdev(v) if len(v) > 1 else None
        out.append(row)
    return out


def paired_group_bootstrap(predictions, seeds, reference="GRU"):
    """Average paired error differences over training seeds, then resample groups."""
    by_split = defaultdict(lambda: defaultdict(dict))
    for r in predictions:
        if r["mode"] == "forced_raw" and r["y"] == "0" and r.get("is_control") not in ("True", "true", "1"):
            by_split[r["split_id"]][r["method"]][r["trial_id"], r["seed"]] = r
    output = []
    for split, methods in by_split.items():
        if reference not in methods:
            continue
        ref = methods[reference]
        for method, records in methods.items():
            if method == reference:
                continue
            candidates = {trial for trial, seed in ref} & {trial for trial, seed in records}
            differences = []
            for trial in sorted(candidates):
                keys = [(trial, str(s)) for s in seeds]
                if not all(k in ref and k in records for k in keys):
                    continue
                values = [int(records[k]["decision"] == "1")-int(ref[k]["decision"] == "1") for k in keys]
                differences.append((records[keys[0]]["group_id"], float(np.mean(values))))
            groups = sorted({g for g, value in differences})
            row = dict(split_id=split, method=method, reference=reference, negative_episode_n=len(differences), physical_group_n=len(groups), paired_seed_n=len(seeds),
                       mean_false_pass_rate_difference=float(np.mean([v for g, v in differences])) if differences else None,
                       ci95_low=None, ci95_high=None,
                       interpretation="configuration-cluster bootstrap of per-episode seed-averaged paired errors; descriptive, selected configurations")
            if len(groups) >= 3:
                chunks = {g: np.asarray([v for group, v in differences if group == g]) for g in groups}
                rng = np.random.default_rng(407099)
                samples = [float(np.concatenate([chunks[g] for g in rng.choice(groups, len(groups), replace=True)]).mean()) for _ in range(1000)]
                row["ci95_low"], row["ci95_high"] = np.quantile(samples, [.025, .975]).tolist()
            elif differences:
                row["interpretation"] += "; CI unavailable: fewer than three negative physical groups"
            else:
                row["interpretation"] = "UNAVAILABLE: missing complete five-seed paired predictions"
            output.append(row)
    return output


def generate(run, report):
    protocol = read_json(run/"frozen_protocol.json")
    queue = read_json(run/"task_queue.json")
    report.mkdir(parents=True, exist_ok=True)
    from learning_models import METHOD_GROUPS, ENGINEERING
    metrics_rows, predictions, bins, evidence = [], [], [], []
    for task in queue["tasks"]:
        folder = run/"fits"/task["task_id"]
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
    all_rows = [r for r in metrics_rows if r["subset"] == "all"]
    schema = ["split_id", "method", "seed", "mode", "n", "passable_n", "blocked_n", "true_pass_n", "true_block_n", "false_pass_n", "false_reject_n", "uncertain_n", "coverage"]
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
        selected = [r for r in all_rows if r["method"] in methods]
        write_csv(report/filename, selected, None if selected else schema)
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
    pairs = paired_group_bootstrap(predictions, protocol["seeds"])
    write_csv(report/"PAIRED_CONFIGURATION_ANALYSIS.csv", pairs, None if pairs else ["split_id", "method", "reference", "negative_episode_n", "physical_group_n", "paired_seed_n", "mean_false_pass_rate_difference", "ci95_low", "ci95_high"])
    write_csv(report/"LEARNING_RUN_STATUS.csv", [{**t, "reasons": "|".join(t["reasons"])} for t in queue["tasks"]],
              None if queue["tasks"] else ["task_id", "split_id", "method", "seed", "status", "reasons"])
    write_csv(report/"LEARNING_ARTIFACT_INDEX.csv", evidence, None if evidence else ["task_id", "completion", "metrics", "metrics_sha256", "predictions_sha256", "checkpoint_sha256"])
    counts = {status: sum(t["status"] == status for t in queue["tasks"]) for status in sorted({t["status"] for t in queue["tasks"]})}
    state = dict(scope=protocol["scope"], evaluated_fit_n=len(evidence), planned_task_n=len(queue["tasks"]), task_counts=counts,
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
        + "`RISK_COVERAGE_POINTS.csv`、`CALIBRATION_BINS.csv`、各fit的calibration.json均来自实际执行。有实际有效bin时生成 `CALIBRATION_RELIABILITY.png`，按固定候选方法展示 clean test 的 raw/temperature 可靠性曲线；汇总权重是预测数，不把重复seed当独立物理样本、不画未经支持的误差带。完整方法及noisy诊断都保留在CSV。只有表头表示尚无实际可用评价，不是零风险。`PAIRED_CONFIGURATION_ANALYSIS.csv`先在同episode上平均五个seed的配对误差，再按物理配置聚类bootstrap；不足3个负配置不报告CI，seed不是独立物理重复。\n", encoding="utf-8")
    report.joinpath("ABLATION_REPORT.md").write_text(
        "# G 消融执行审计\n\n"
        + f"真实已评价fit：{len(evidence)}；详细未运行/受阻原因见 `LEARNING_RUN_STATUS.csv`。当前范围 `{protocol['scope']}`，不据模型数量宣布完整benchmark完成。\n\n"
        + "M2候选为标准监督对比或有证据的物理配对cosine contrastive，非原创对比算法。完整匹配及两个消融要求同一个训练集同时具有真实困难负对照和跨配置正对照；缺失则全部明确受阻，不复制/镜像标签。离线元数据不进入特征。\n\n"
        + "M3包括真实观测历史、固定seed乱序、删累计手部进展的末观测、命令进程、真实dt+进程、固定.5/1s前缀及按结构参数数匹配的GRU。只保留真实观测；高密度数据最多128点，用固定时间网格的past索引去重，低频日志不插值造高频。参数匹配不使用标签/测试。\n\n"
        + "同episode的GRU10Hz/GRU20Hz使用相同2s截止及真实past观测去重，保留安全早停；基准实际频率以源数据审计为准，不以模型名称假定50Hz。所有方法都可从已获得样本长度推知提前停止，属于合法但可能依赖安全策略的线索；DurationOnlyLogistic仅用已获得点数/观测时长作该捷径诊断。GRUShuffled与普通GRU同预算通道，均删除显式elapsed/dt/command_progress；已执行q_cmd本身仍可携带阶段线索，乱序不能单独证明无时序信息。\n\n"
        + "M4联合响应辅助使用训练组（含UNKNOWN）的下一已记录q/qdot；分阶段版本40轮响应预训练+80轮监督微调，普通/联合为80轮，因此额外训练预算必须披露，不作纯损失因果结论。任何测试/校准UNKNOWN不进入拟合或归一化。\n\n"
        + "M5按真实关节名称重排active/opposite手并保留侧别，使用既有body-local FK；它不是经过运动学认证的带符号几何镜像。没有镜像标签或强制左右动作同标签。OldM2去FiLM的残差参照仍通过正常响应预测器间接使用身体状态，不是完全身体信息剥离。\n\n"
        + "观测DR有真实执行入口：GRUObsDR与普通GRU同结构/同clean训练scaler，只对训练原始观察作固定seed扰动；所有模型另用同一固定seed的noisy heldout诊断，校准概率始终保持clean。q=.001rad、qdot=.01rad/s、effort=训练尺度2%噪声/1%每条偏置、IMU角速度=.005rad/s/姿态=.001rad、时钟±.002s、丢样1%；延迟按真实50Hz可实现的0/20ms端点逐episode抽取，严格不插值。真实source年龄不得超过20ms；稀疏日志无法实现时单列不可用。尺度仅训练数据，元数据/标签不变；它是未硬件标定的仿真敏感性，并未耦合重算FK。source/receipt索引、seed、每条偏置与时钟保留audit；实际运行前结果表为空，不以实现代码宣称DR实验已完成。\n\n"
        + "需要从实际CSV解释优势/无优势；本生成器不会以尚未运行的代码或空表给方法有效性结论。\n", encoding="utf-8")
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
