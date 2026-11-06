"""Read-only history/protocol audit for G; preservation/git audit is separate."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/round4070_comprehensive_traversability'
REPORT=ROOT/'research/round4070_comprehensive_traversability'
CAMPAIGNS=['g1_final4070','round4070_problem_discovery','round4070_failure_discovery','round4070_structural_probe_discovery','round4070_fixed10cm_judgment','round4070_fixed_probe_benchmark','round4070_learning_traversability']
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return p.relative_to(ROOT).as_posix()


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    docs=[]
    for campaign in CAMPAIGNS:
        for p in sorted((ROOT/'research'/campaign).rglob('*')):
            if p.suffix.lower() not in {'.md','.csv','.json','.txt'}:continue
            text=p.read_text(encoding='utf-8-sig')
            docs.append(dict(path=rel(p),sha256=sha(p),characters=len(text),lines=len(text.splitlines()),
                protocol_evidence=[s for s in text.splitlines() if re.search('PROBE_|UNKNOWN|runtime.valid|probe.valid|clearance|10.Hz|200.Hz|标签|协议',s,re.I)][:30]))
    old=load(ROOT/'outputs/round4070_learning_traversability/dataset/historical_inventory.json')
    source_errors=[];history_counts=[]
    for campaign in old:
        phase=Counter();sizes=[]
        for h in campaign['histories']:
            p=ROOT/h['path']
            if sha(p)!=h['sha256']:source_errors.append(rel(p))
            with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            phase.update(r.get('phase','') for r in rows);sizes.append(len(rows))
        history_counts.append(dict(campaign=campaign['campaign'],actual_histories=len(sizes),actual_rows=sum(sizes),phases=dict(phase)))
    datasets={}
    for name in ('dataset','dataset_phase_c','dataset_final'):
        p=ROOT/'outputs/round4070_learning_traversability'/name/'manifest.json';m=load(p)
        samples=0;dts=[];source_files=0;hash_errors=[]
        for e in m['episodes']:
            with np.load(ROOT/e['npz_path'],allow_pickle=False) as d:
                seq,t=d['seq'],d['time'];names=d['feature_names'].tolist()
                assert seq.shape==(len(t),166) and np.isfinite(seq).all() and np.all(np.diff(t)>0)
                assert names==m['feature_names']
                samples+=len(t);dts.extend(np.diff(t).tolist())
            if name=='dataset':
                # The raw path is stored in E's original frozen campaign record.
                pass
        core=[e for e in m['episodes'] if not e['is_control']]
        datasets[name]=dict(manifest=rel(p),manifest_sha256=sha(p),episodes=len(m['episodes']),samples=samples,
            core_labels=dict(Counter(e['label'] for e in core)),control_labels=dict(Counter(e['label'] for e in m['episodes'] if e['is_control'])),
            positive_hands=dict(Counter(e['side'] for e in core if e['y']==1)),positive_physical_groups=sorted({e['group_id'] for e in core if e['y']==1}),
            actual_median_period_s=float(np.median(dts)),feature_count=len(m['feature_names']))
    prediction_path=ROOT/'outputs/round4070_learning_traversability/phase_c/predictions.csv'
    with prediction_path.open(encoding='utf-8-sig',newline='') as f:pred=list(csv.DictReader(f))
    results=[]
    for method in ('Logistic','GRU','TCN','ResidualGRU','M2','GRUShuffled','Instantaneous'):
        for seed in ('4070','4071','4072'):
            rows=[r for r in pred if r['method']==method and r['seed']==seed and r['fold'] in {'family_fold_0','family_fold_1'}]
            assert len(rows)==36 and len({r['trial_id'] for r in rows})==36
            p=np.array([float(r['p']) for r in rows]);y=np.array([int(r['y']) for r in rows]);d=p>=.5
            results.append(dict(method=method,seed=int(seed),n=len(rows),P=int(sum(y==1)),B=int(sum(y==0)),FP=int(sum(d&(y==0))),FN=int(sum(~d&(y==1)))))
    audited=dict(status='PASS' if not source_errors else 'HISTORICAL_SOURCE_HASH_MISMATCH',reports_read=docs,
        old_histories=history_counts,historical_hash_mismatches=source_errors,F_dataset_snapshots=datasets,phase_c_recomputed_metrics=results,
        phase_c_predictions_sha256=sha(prediction_path),
        key_correction='Legacy local+x bottom_right rotated yaw90 is world+Y robot-left. E/F bottom_right physically maps to G B02, never B03 by name.',
        legacy_supervision='Not imported into G: old clearance/action-selection semantics not proven equivalent to fixed G action intervention')
    (OUT/'historical_protocol_audit.json').write_text(json.dumps(audited,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    text='''# 历史协议、输入与标签复核（Round4070-G）

旧文件全部只读；源代码、报告和冻结产物的保全哈希由独立 preservation 审计负责。这里重新读取七个历史目录的全部文本报告/表格，逐份核对136份旧 CSV 与 F 保存的原始 SHA-256，再读取 E104、Phase C115、最终117份合法 NPZ，复算最终 Phase C 逐样本预测计数。实际路径、文本摘录、哈希与统计在 `outputs/round4070_comprehensive_traversability/historical_protocol_audit.json`。

| 历史范围 | 实际完整历史 | 协议/标签语义 | G用途 |
|---|---:|---|---|
| Round4070原始 | 24 | 探测、后续通过与导航阶段并存 | 审计，不能把后续T2当probe输入 |
| problem discovery | 37 | 38启动/37完整；14 clearance并不使其余成为BLOCKED | 异协议单列 |
| B failure discovery | 38 | 42启动/38有效；不同深度、左右手反例与隐藏板 | 异协议单列 |
| C structural | 27 | 30启动/27有效；结构和探测深度同时变化；1非控制clearance | 不能直接与固定探测混合 |
| D fixed10cm | 10 | 8运行＋2独立全臂校准；核心正式P/B均0 | 未完成/安全撤回不是负标签 |
| E benchmark | 104 | 单次名义10cm，允许contact-limited；核心P2/B26/U60，控制16 | 原F数据来源，G不继承旧监督 |
| F最终 | 117 | E104＋F13；核心P7/B29/U65，控制P14/B2 | 旧标签仅审计；合法训练组历史可另行响应适配 |

**采样错误的精确范围。** physics=200Hz，FALCON控制=50Hz；旧CSV仅常规10Hz，20–22点/约2s。E在线 `deployment_samples` 每个physics step追加，因此在线规则 DT=.005正确。错误位于离线 `evaluate_benchmark.py` 把10Hz CSV按.005累计stall；真实时间修正保留原冻结阈值，E1正确BLOCKED由0升到8/26，E2由0升到4/26。不能改在线DT为.1，也不能把修复错误算神经网络创新。G导出50Hz真实采样并按 `sum(diff(time)*low_speed[:-1])` 计时，无首尾外推/插值。

**输入与时间语义。** q/qdot/q_cmd全部官方29关节顺序，q_cmd确已下发后才采集反馈；effort来自applied_torque仿真执行器proxy，并非经真机校准的力/电流。旧previous_action在compute_target后读取，实际是当次policy action；10Hz上一日志行不是50Hz上一控制动作。旧F仅保存gyro_z、没有gyro_x/y与加速度；新G在compute前复制真正previous_control_action，完整gyro/重力与body状态合法保存。G保留加速度候选供独立smoke核验，验证前模型通道明确缺失。关节左右映射按名称，未验证的带符号镜像不可擅用。

**左右约束命名纠正。** 旧E/F `bottom_right_index`是布的local+x，yaw=90°后落在world+Y；机器人朝+X时这是robot-left。新G B02为world+Y左下角，物理上对应旧FABRIC_BOTTOM_RIGHT_ANCHOR；B03对应旧BOTTOM_LEFT。不能按名称直接等价，必须核对实际固定节点/连接对象/世界位置。`top-edge + bottom two corners`与`four-corners-only`也始终分开。

**冻结学习证据。** Phase C用115条快照、两个家族fold，36个正式测试标签每method/seed唯一；训练数19/28。最终追加2条左手均UNKNOWN，因此最终数据117但未重训原115模型。7个正例仍全右手，仅4个完整物理配置、两种接近结构家族和1.95/2.00m尺寸；负例主要隐藏刚板。普通GRU FP=3/29（三seed均3），残差2/3/3，完整M2 3/6/3，shuffled GRU 3/3/3；没有时序顺序或复杂M2独立收益证据。F冻结8有效标签上连Logistic也全对，不能把该小样本当独立方法优势。

**新契约不自动继承旧真值。** G固定PASSABLE_FOR_ACTION及独立于分类器选择的后续动作。旧E/F通过条件、动作选择、恢复与实际执行范围不自动与G等价。G importer将旧117条全部标为响应源y=-1，维持原166维与缺失字段；不把旧P/B移入252维G监督，不合成未记录传感器。若未来适配旧响应历史，仍需按真实物理等价组排除G测试组。
'''
    (REPORT/'HISTORICAL_PROTOCOL_AUDIT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(status=audited['status'],text_reports=len(docs),old_histories=sum(x['actual_histories'] for x in history_counts),
                         snapshots=datasets,phase_c_metrics=results),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
