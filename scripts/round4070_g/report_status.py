"""Write a candid research checkpoint from actual ledgers, never planned scores."""
from collections import Counter
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_comprehensive_traversability'
DOC = ROOT / 'research/round4070_comprehensive_traversability'


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return default


def main():
    progress = read(DOC / 'PROGRESS_CHECKPOINT.json', {})
    dataset = read(OUT / 'dataset/manifest.json', {})
    campaign = read(ROOT / progress['frozen_manifest'], {}) if progress.get('frozen_manifest') else {}
    primary_ids = {p['trial_id'] for p in campaign.get('trial_plans', [])}
    episodes = [e for e in dataset.get('episodes', []) if e.get('planned_trial_id',e.get('trial_id','')) in primary_ids]
    core = [e for e in episodes if not e.get('is_control')]
    with (DOC / 'SCENE_COVERAGE_MATRIX.csv').open(encoding='utf-8-sig', newline='') as handle:
        scenes = list(csv.DictReader(handle))
    required = [s for s in scenes if not s['scene_id'].startswith('DR') and s['scene_id'] != 'EMPTY']
    valid = [s['scene_id'] for s in required if s['physics_valid'].lower() == 'true']
    labelled_scenes = sorted({e['scene_id'] for e in core if e['y'] >= 0})
    latest = read(OUT / 'latest_learning_run.json', {})
    learning = read(ROOT / latest['run'] / 'task_queue.json', {}) if latest else {}
    tasks = learning.get('tasks', [])
    full_tasks=[t for t in tasks if t.get('evaluation_scope')!='PARTIAL_HELDOUT_SINGLE_CLASS']
    partial_tasks=[t for t in tasks if t.get('evaluation_scope')=='PARTIAL_HELDOUT_SINGLE_CLASS']
    counts = dict(Counter(t['status'] for t in full_tasks))
    evaluated = sum(t['status'] == 'EVALUATED' for t in full_tasks)
    partial_evaluated=sum(t['status']=='EVALUATED' for t in partial_tasks)
    header = ('# Round 4070-G 阶段报告：研究未完成\n\n'
        f"实际检查时间：`{progress.get('checked_utc', '未生成')}`。当前状态 **IN_PROGRESS_NOT_COMPLETE**。"
        '这不是终局完成报告，未满足用户定义的任何完整 benchmark 完成条件。计划、实现、合成代码验证、真实仿真、正式学习评价分开记录。'
        f"当前冻结 manifest 为 `{Path(progress.get('frozen_manifest','')).name}`，primary protocol 为 `{campaign.get('g1_primary_protocol','PENDING')}`。"
        '没有明确 seed 随机化、完整 G1 碰撞证据及匹配动作契约的旧 pilot 排除出主实验；原始 C01 PASSABLE 不再作为主监督正例。'
        'V1 有限墙标签缺口见 ACTION_LABEL_GAP_REPORT.md；实际新版动作范围以当前冻结 sidecar 为准，不能追溯改写旧试验。\n\n')
    inventory = (f"已实际启动 physics {progress.get('physics_attempts_launched', 0)} 次、G1 {progress.get('g1_attempts_launched', 0)} 次；"
        f"三独立 seed 门禁通过配置 {progress.get('physical_configs_with_three_valid_seeds', 0)} 个（含控制/组合）。"
        f"指定 48 个命名场景中有 {len(valid)} 个具备三 seed 门禁证据；相同物理配置的别名不算独立验证。\n\n"
        f"主实验导入 {len(episodes)} 条有效探测历史，其中空场控制 {len(episodes)-len(core)} 条，核心标签 `{dict(Counter(e['label'] for e in core))}`。"
        f"另有 {progress.get('superseded_pilot_attempts_launched',0)} 次已启动旧 pilot，仅保留诊断，不进入主训练或统计。"
        f"当前冻结主学习状态 `{counts}`，实际主 EVALUATED 任务 {evaluated} 个；独立单类诊断 EVALUATED {partial_evaluated} 个。合成 smoke 不进入这些计数。\n\n")
    answers = [
        ('全部场景家族物理验证', f'尚未完成。三 seed 通过的指定命名场景为 {", ".join(valid) or "无"}；逐项状态及失败证据见 SCENE_COVERAGE_MATRIX.csv 和 PHYSICAL_VALIDITY_REPORT.md。'),
        ('产生 G1 有效通行标签的场景', f'核心场景：{", ".join(labelled_scenes) or "暂无"}。控制样本单列；超时、安全撤回、PointGoal 不作为核心通过标签。'),
        ('材料可信度', '布、网孔和 PVC/条带均为当前求解器中的物理代理；真实材料参数、碰撞接触厚度、弯折/塑性尚未经过实物校准。数值门禁不证明材料保真，材料 OOD 不据此宣称实物泛化。'),
        ('左右手正负覆盖', f'尚需按 LEFT_RIGHT_COVERAGE.csv 核对 50 P/50 B、每手正负及至少三个独立正例家族，当前核心总数 {len(core)}。不以重试或物理别名扩充独立家族。'),
        ('Domain randomization 范围', '已冻结几何/约束/动力学、机器人偏移/姿态/动作时长与观测扰动。数值范围、12 个 LHS 组合、实际逐 seed 初态注入及运行状态见 DOMAIN_RANDOMIZATION_SPEC.md、CURRENT_CAMPAIGN.json 指向的冻结 manifest 和执行表。独立初始预张力尚未实现；观测扰动为未硬件标定的敏感性试验。'),
        ('普通 GRU 是否最强', '尚无完整 G 独立评价支持该排序；F 的结果不可替代新动作合同。'),
        ('LSTM、TCN、Transformer、残差 GRU', '方法实现及合成 smoke 不是科学比较；相对表现以同一冻结划分的五训练 seed 实际结果为准，目前不作排名。'),
        ('匹配对比学习跨约束泛化', '必须有真实跨配置正对与同观测条件硬负对、匹配消融和独立固定方式测试；缺少合法配对时任务明确阻塞，不构造镜像标签。当前不能声称改善。'),
        ('命令/进程条件化与时间顺序', '需要完整历史、乱序、末观测、命令/进程、时长捷径及固定前缀对照。q_cmd 本身可能携带阶段信息，乱序对照不能单独证明时序无用。当前没有充分结果。'),
        ('无标签响应预测与标签效率', '仅训练物理组的 UNKNOWN 可参与；需同标签预算独立测试。分阶段预训练额外 40 epoch 必须单列成本，目前不能声称提高效率。'),
        ('跨手与姿态稳定性', '双手空场 smoke 用于验证日志与动作执行，不能代替跨手分类泛化。按 active/opposite 关节重排不是已认证的带符号运动学镜像；需双向跨手和姿态 OOD 评价。'),
        ('同误放行风险下覆盖提升', '仅用独立 calibration 选择阈值，在未触碰的 test 统计风险与覆盖；有限样本经验风险不等于保证。尚无可重复优势结论。'),
        ('材料/隐藏墙记忆', '物理组、固定方式、组合及材料留出用于检查。由于材料保真未验证和比较未完成，目前不能排除捷径或认定已解决。'),
        ('同信息下不可可靠区分', '远处 D07 不在当前探测范围，标签保持 UNKNOWN 并要求新视觉/深度观测；未接触、控制拒绝或多个物理条件产生相似本体响应也可能不可辨识，需实际匹配证据，不能把 UNKNOWN 训练成负例。'),
        ('RA-L/T-RO 方法贡献', '现阶段证据不足。标准 GRU、残差、对比学习和选择性分类不是原创；潜在价值在可信干预标签、约束/动作条件化和独立泛化证据，必须由后续结果支持。'),
    ]
    lines = '\n'.join(f'{i}. **{title}**：{answer}\n' for i, (title, answer) in enumerate(answers, 1))
    tail = ('\n完整待办仍包括未通过或未执行的物理门禁、核心 G1 双手覆盖、确认集、正式五 seed 方法比较、消融及独立 OOD/校准。'
        '当前材料保真与独立预张力限制不能靠更换 GPU 消除。\n\n'
        '仅单类有支持的留出集可另列 PARTIAL_HELDOUT_SINGLE_CLASS 诊断，报告该类实际计数；它不替代双类 benchmark，不支持 balanced accuracy、AUROC 或完整 OOD 优势结论。\n\n'
        '恢复命令：`powershell -NoProfile -ExecutionPolicy Bypass -File .\\run_round4070_g.ps1 -Mode all -RetryRuntimeFailures`。'
        '它从持久队列继续实际执行，不代表聊天结束后代理会自动继续审查。详细操作见 RUN_INSTRUCTIONS.md；Git 记录见 GIT_DELIVERY_REPORT.md。\n')
    (DOC / 'FINAL_REPORT_ZH.md').write_text(header + inventory + lines + tail, encoding='utf-8')
    (DOC / 'SCIENTIFIC_METHOD_DECISION.md').write_text('# Scientific decision at the actual checkpoint\n\n'
        'Status: **IN_PROGRESS_NOT_COMPLETE**. No method winner or method contribution is selected.\n\n'
        + inventory + 'Do not assign LEARNING_EVIDENCE_INSUFFICIENT as a substitute for unrun learning: that user-defined status requires completed model training. '
        'Do not assign NO_METHOD_ADVANTAGE without the comparisons. METHOD_ADVANTAGE_SUPPORTED requires repeated independent evidence at matched false-pass risk and a valid calibration/test separation. '
        'FULL_BENCHMARK_COMPLETE and COMPLETE_WITH_SCENE_LIMITATIONS require all executable planned comparisons, not merely implementation or queue creation.\n\n'
        'The decision remains open until independent physical labels, all standard baselines, candidate ablations, cross-hand/pose/constraint tests and the confirmation set are actually evaluated. '
        'Physical aliases, stochastic training seeds, augmented observation copies, synthetic smoke cases and empty controls do not create new independent positive obstacle families.\n', encoding='utf-8')
    print(json.dumps(dict(status='IN_PROGRESS_NOT_COMPLETE', required_scenes_three_seed_valid=len(valid), core_histories=len(core), evaluated_tasks=evaluated)))


if __name__ == '__main__':
    main()
