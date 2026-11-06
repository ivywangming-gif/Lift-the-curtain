"""Summarize actual F physics artifacts, keeping absent evidence explicit."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from physics_queue import OUT, REPO, read, write, summarize

summarize()
rows=read(OUT/'supplementary_labels.json')
prior=read(OUT/'supplementary_labels_before_left_completion.json')
if prior:assert rows[:len(prior)]==prior, 'Previously frozen supplementary labels were modified'
g1=[]
for p in sorted((OUT/'trials').glob('*/trial_metadata.json')):
    root=p.parent; s=read(root/'summary.json');w=read(root/'watchdog_summary.json');r=read(root/'resource_summary.json')
    g1.append(dict(trial=root.name,runtime_valid=bool(s.get('runtime_valid')),probe_valid=bool(s.get('probe_valid')),summary_exists=bool(s),stage=w.get('last_valid_stage','STOPPED_BEFORE_SUMMARY'),timeout=w.get('timeout_reason'),label=next((x['label'] for x in rows if x['trial_id']==root.name),'UNKNOWN'),gpu_peak_mb=r.get('gpu_memory_peak_mb'),ram_peak_mb=r.get('process_private_peak_mb')))
p1=[]
for p in sorted((OUT/'p1').glob('*/*/*/summary.json')):
    s=read(p)
    penetration={}
    snapshots=p.parent/'runtime_node_snapshots.npz'
    if p.parent.name=='probe' and snapshots.exists():
        with np.load(snapshots) as data:
            centers=data['probe_positions']
            arrays=[data[k] for k in data.files if k.startswith('runtime_positions__')]
            if len(centers) and arrays:
                nodes=np.concatenate(arrays,axis=1)
                # 0.12 m cube with 5 mm tolerance. This necessary node test
                # cannot detect a triangle crossing between sampled vertices.
                interior=(np.abs(nodes-centers[:,None,:]) < .055).all(axis=-1)
                penetration=dict(sampled_nodes_inside_probe_max=int(interior.sum(axis=1).max()),sampled_nodes_inside_released_probe=int(interior[-1].sum()),test_scope='additional node test; see runtime extended geometry audit for triangle test')
    extended={k:v for k,v in s.get('extended_geometry_audit',{}).items() if k!='rail_samples'}
    rail=s.get('extended_geometry_audit',{}).get('rail_samples',[])
    if rail:extended['rail_maxima']={key:max(abs(r[key]) for r in rail) for key in ['dy_m','transverse_drift_m','relative_anchor_drift_m','rotation_error_rad']}
    p1.append(dict(scene=p.parents[2].name,seed=p.parents[1].name,mode=p.parent.name,status=s.get('status'),runtime_status=s.get('runtime_status'),max_speed_mps=s.get('max_speed_mps'),failed_acceptance=[k for k,v in s.get('acceptance',{}).items() if not v],additional_validity_required=s.get('additional_validity_required',[]),error=s.get('error'),probe_node_audit=penetration,extended_geometry_audit=extended))
p0=[]
for p in sorted((OUT/'p0_gate').glob('*/*/summary.json')):
    s=read(p);p0.append(dict(condition=p.parents[1].name,mode=p.parent.name,status=s.get('status'),error=s.get('error')))
summary=dict(g1_attempts=len(g1),runtime_valid=sum(x['runtime_valid'] for x in g1),probe_valid=sum(x['probe_valid'] for x in g1),labels=dict(Counter(x['label'] for x in rows)),g1=g1,p1=p1,p0_new_size=p0)
summary['prior_11_labels_unchanged']=bool(prior and rows[:len(prior)]==prior)
summary['withheld']=[dict(path=str(p.relative_to(OUT)),**read(p)) for p in sorted(OUT.rglob('withheld.json'))]
for item in summary['withheld']:
    path=OUT/item['path']
    prior_init=path.parent.parent/'no_probe'
    stderr=prior_init/'process_stderr.log'
    if not (prior_init/'summary.json').exists() and stderr.exists() and 'boundary pause exceeded' in stderr.read_text(encoding='utf-8',errors='replace'):
        item['raw_status']=item.get('status')
        item['status']='WITHHELD_EXECUTION_INVALID'
        item['reason']='Training boundary pause expired before Isaac startup; no physical initialization was attempted'
summary['p1_execution_invalid']=[dict(path=str(p.parent.relative_to(OUT)),reason='TRAINING_BOUNDARY_PAUSE_TIMEOUT' if 'boundary pause exceeded' in (p.parent/'process_stderr.log').read_text(encoding='utf-8',errors='replace') else 'NO_SUMMARY_EXECUTION_INVALID') for p in sorted((OUT/'p1').glob('*/*/*/command.json')) if not (p.parent/'summary.json').exists() and (p.parent/'process_stderr.log').exists()]
write(OUT/'PHYSICS_SUMMARY.json',summary)
lines=['# Round 4070-F 物理有效性与补充采集','',
'本报告由真实文件生成；候选清单不等于完成。日志根：`outputs/round4070_learning_traversability/physics/`。',
'',f"G1 实际启动目录 {len(g1)}；runtime-valid {summary['runtime_valid']}；probe-valid {summary['probe_valid']}；有 summary 的标签分布 {summary['labels']}。无 summary 为 UNKNOWN/execution-invalid，不当作 BLOCKED。",'',
'## G1 实际结果','', '| trial | runtime | probe | label | last stage / timeout | GPU peak MB |','|---|---:|---:|---|---|---:|']
for x in g1:lines.append(f"| {x['trial']} | {x['runtime_valid']} | {x['probe_valid']} | {x['label']} | {x['stage']} / {x['timeout']} | {x['gpu_peak_mb']} |")
lines+=['','原沙箱启动卡在 BEFORE_APP，第一条达到既定 180 s watchdog；第二条在诊断时停止队列，进程树均已退出。通过受审查的 require_escalated，以相同 seed/动作/超时在独立 `_UNSANDBOXED_RETRY` 目录重试。未覆盖旧失败记录、未放宽 startup gate、未升级环境。','',
'`CLEANUP_TIMEOUT_AFTER_FINALIZE_30s` 发生在完整summary及FINALIZE写出之后，是既有应用退出清理窗口，不是OOM/CUDA/物理求解失败；guard只清理本次启动PID。运行/探测有效性仍由完整实际数据独立审计。','',
'后续标名 `CONFIRM` 的 1.95 m 试验是预先登记的新尺寸探索，不是同条件独立确认；screen 2.00 m 与新尺寸不能混称确认。姿态扩展是预注册的第二个近邻初始姿态，只在有真实正例后启动。姿态与 seed 同时改变，不是同 seed 的姿态因果干预，无法隔离种子/求解随机性。UNKNOWN 没有监督真值，UNKNOWN→PASSABLE 不能称为已确认通行标签翻转。','',
'原11条完成后另行冻结两条1.95m左手补充（420401/420402），只为填补左手正例空白，不依据模型误判选择条件；无论结果如何仅两条，不追正例。复用双physics gates与已验证left动作安全契约，先冻结模型评价。原11行JSON完整保留，最终脚本断言原行内容与顺序不变后才写报告。','',
'## 新尺寸 physics-only','', '| condition | mode | status |','|---|---|---|']
for x in p0:lines.append(f"| {x['condition']} | {x['mode']} | {x['status']} |")
lines+=['','## P1 六类真实尝试','', '| scene | seed | mode | runtime | sanity gate | failed checks |','|---|---|---|---|---|---|']
for x in p1:lines.append(f"| {x['scene']} | {x['seed']} | {x['mode']} | {x['runtime_status']} | {x['status']} | {', '.join(x['failed_acceptance']) or x.get('error') or 'none'} |")
lines+=['',f"新尺寸P0实际完成 {len(p0)} 次physics-only；P1六类共24次，再加mobility一次共 {len(p1)} 次，每次720步/3.6 s；属于有限重复，不是长期稳定性或材料标定。",'', '无探测初始化不通过时，后续 probe/G1 不启动；以下为真实未运行记录，不计入 completed：','']
lines += [f"- `{x['path']}`：{x['status'] if 'status' in x else x.get('reason')}" for x in summary['withheld']]
lines += ['','P1 另列 execution-invalid（不计 physics-invalid，不是材料负例）：',''] + [f"- `{x['path']}`：{x['reason']}" for x in summary['p1_execution_invalid']]
if summary['p1_execution_invalid']:lines += ['','PLEATED_SINGLE 首入口在 CPU 训练边界等待时达到360 s上限，尚未导入/启动Isaac，后续以新seed/目录实际重试。该入口的旧withheld文件通用PHYSICS_INVALID字样应按此启动审计纠正为EXECUTION_INVALID，原记录未覆盖。']
lines+=['','P1 全部为 physics-only，无机器人、无通行标签。新增检查每 0.005 s 记录实际 probe pose（修正旧待发 target 的一步偏差），读取实际 USD mesh faces 做 triangle–box SAT 深穿透检查（120 mm 探针、5 mm容差），撤回末尾验证几何释放及节点速度。滑轨以只读 RigidObject 包装现有 dynamic support，记录真实 PhysX pose，检查 ±0.30 m rail（5 mm容差）、X/Z 10 mm、旋转0.01 rad，以及顶部节点相对 moving support 的10 mm附着漂移。上述检查只有前置初始化通过才推进probe；结果并入实际 acceptance 列，原始pose保存在summary。',
'', '离散检查不等于连续时间无穿透证明，也不等于真实材料模型标定；`p1_eligible_for_g1=false` 表示本轮未建立 P1 G1 scene integration，不能冒称已生成 G1 标签。数值门和已检查几何机制的通过/失败在上表独立报告。',
'', '滑轨移动的是整根顶部support，不是独立窗帘环的滑动聚拢。原沿X探针只产生微米量级rail Y位移，不证明滑动开帘能力；另列 SLIDING_TOP_TRACK_MOBILITY 直接施加 world-Y 0.5 N、0.5 s小力，验证模型自由度与限位，不是G1手部动作或通行标签。',
'', '网状候选采用删除棋盘式内部面片的几何 proxy，可能形成仅顶点相连的非流形网格；它不是已标定蚊帐。PVC 条带是三个独立 surface-deformable sheets，不是 articulated surrogate，不是真实 PVC constitutive model。真实 PVC/蚊帐的材料有效性状态为 `PHYSICS_INVALID`（所需真实材料模型尚未成立）；sanity 仅描述代理几何的数值行为。',
'','## 历史覆盖、冻结与界限','',
'已读取 4070/B/C/D/E 实际报告和 E summary。E nominal/right 与 bottom-right/right 各一个正式 clearance 正例，是本轮优先复验来源。E 重叠多片的初始数值速率超门限；其部分 summary 错写 articulated，实际代码为 surface deformable，本轮已纠正模型语义。',
'','P0 既有顶部整边、整边加单/双底角、真正四角、全周边、重自由/受限、2/5/8 cm 后板与控制场景只按原有证据继承。新尺寸各自先做无探测/探测 gate。真正三角固定未在 F 构造；顶部整边加底角及 P1 双片单侧约束均不冒称三角固定。',
'','G1 继承 gate 启动前读取真实双 summary 并存 SHA256；新尺寸 gate 不通过便不启动该 G1。冻结 FALCON SHA256 为 `8ac8f51875b878a79d9b5782e702b66572697e204ed262e2002b55631f3105d0`，代码 assert 验证。单环境串行、headless、render=False、heartbeat/watchdog、GPU/RAM/CPU telemetry；未修改任何旧 trial。',
'','探测 0.10 m 为名义命令；实际位移不足不终止全局实验。安全停止、probe-valid、独立 clearance 标签分开记录。新标签沿用 E 的 protocol_audit/independent_label；UNKNOWN 不作负例。',
'','复现：`C:/conda_envs/isaaclab/python.exe scripts/round4070_f/physics_queue.py --stage screen --retry-tag NEW_RUN_ID`；其后依阶段证据运行 `--stage confirm` / `--stage posture` / `--stage p1`，已有输出绝不覆盖。最终运行 `physics_finalize.py`。新 ID 不代表新物理配置；完整配置仍应分组。']
mobility=next((x for x in p1 if x['scene']=='SLIDING_TOP_TRACK_MOBILITY'),None)
if mobility and mobility['extended_geometry_audit'].get('rail_maxima'):
    maxima=mobility['extended_geometry_audit']['rail_maxima']
    at=lines.index('## 历史覆盖、冻结与界限')
    lines[at:at]=[f"mobility实际结果：{mobility['status']}；最大Y行程 {maxima['dy_m']:.6f} m，相对support锚定漂移最大 {maxima['relative_anchor_drift_m']:.6f} m；没有增加试次或提高力来追求通过。",'']
(REPO/'research/round4070_learning_traversability/PHYSICS_VALIDITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['g1','p1','p0_new_size']},ensure_ascii=False))
