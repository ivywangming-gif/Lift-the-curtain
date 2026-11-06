"""Exploratory grouped proprioceptive learning; never trains on UNKNOWN labels.

No sklearn dependency or simulator import. All normalization/normal-response
fitting uses training groups only. Fixed architectures, no test-driven tuning.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4070_learning_traversability'
REPORT = ROOT / 'research/round4070_learning_traversability'
torch.set_num_threads(2)

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, sorted(set().union(*(r.keys() for r in rows))))
        w.writeheader(); w.writerows(rows)

def metrics(y, p, low=.5, high=.5):
    y, p = np.asarray(y), np.asarray(p)
    d = np.where(p >= high, 1, np.where(p < low, 0, -1))
    pos, neg = y == 1, y == 0
    tp, tn = int(((d == 1) & pos).sum()), int(((d == 0) & neg).sum())
    fp, fn, un = int(((d == 1) & neg).sum()), int(((d == 0) & pos).sum()), int((d < 0).sum())
    P, N = int(pos.sum()), int(neg.sum())
    pr, br = tp/P if P else None, tn/N if N else None
    return dict(n=len(y), passable_n=P, blocked_n=N, true_pass_n=tp, true_block_n=tn,
                false_pass_n=fp, false_reject_n=fn, uncertain_n=un,
                false_pass=fp/N if N else None, false_reject=fn/P if P else None,
                passable_recall=pr, blocked_recall=br, coverage=1-un/len(y) if len(y) else None,
                uncertain_rate=un/len(y) if len(y) else None,
                balanced_accuracy=(pr+br)/2 if P and N else None,
                brier=float(np.mean((p-y)**2)) if len(y) else None)

def resample(x, n=48, times=None, duration=None, fraction=1.):
    if times is None: times=np.arange(len(x),dtype=float)
    t=np.asarray(times)-times[0]
    stop=t[-1] if duration is None else min(float(duration),t[-1])
    if fraction!=1.: raise ValueError('Use a fixed physical duration, not future episode length')
    query=np.linspace(0,stop,n)
    # Past hold rather than interpolation using a future response.
    return x[np.maximum(0,np.searchsorted(t,query,side='right')-1)]

def stat(x):
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)

class StumpBoost:
    """Bernoulli gradient boosting, depth-one regression trees, fixed 60 stages."""
    def fit(self, x, y):
        self.bias = float(np.log((y.sum()+.5)/(len(y)-y.sum()+.5)))
        f = np.full(len(y), self.bias); self.trees=[]
        cuts = [(j, float(t)) for j in range(x.shape[1])
                for t in np.unique(np.quantile(x[:, j], [.2,.4,.6,.8]))]
        for _ in range(60):
            p=1/(1+np.exp(-np.clip(f,-25,25))); residual=y-p
            best=None
            for j,t in cuts:
                m=x[:,j]<=t
                if m.sum()<2 or (~m).sum()<2: continue
                a,b=float(residual[m].mean()),float(residual[~m].mean())
                loss=float(((residual-np.where(m,a,b))**2).sum())
                if best is None or loss<best[0]: best=(loss,j,t,a,b)
            if best is None: break
            _,j,t,a,b=best; self.trees.append((j,t,a,b)); f+=.2*np.where(x[:,j]<=t,a,b)
        return self
    def predict(self,x):
        f=np.full(len(x),self.bias)
        for j,t,a,b in self.trees: f+=.2*np.where(x[:,j]<=t,a,b)
        return 1/(1+np.exp(-np.clip(f,-25,25)))

class Temporal(nn.Module):
    def __init__(self, kind, f, arm, body, ntime):
        super().__init__(); self.kind=kind; self.arm=arm; self.body=body
        if kind in ['Logistic','StatsMLP','HistoryMLP','SingleFrame','Orderless']:
            size=f*4 if kind in ['Logistic','StatsMLP'] else (f*ntime if kind=='HistoryMLP' else f)
            self.net=nn.Linear(size,1) if kind=='Logistic' else nn.Sequential(nn.Linear(size,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
        elif kind=='TCN':
            self.conv=nn.Sequential(nn.Conv1d(f,32,3,padding=2),nn.ReLU(),nn.Conv1d(32,32,3,dilation=2,padding=4),nn.ReLU())
            self.head=nn.Linear(32,1)
        elif kind in ['M2','M2NoResidual','M2Orderless']:
            self.a=nn.GRU(len(arm),64,batch_first=True)
            self.b=nn.Sequential(nn.Linear(len(body),32),nn.ReLU(),nn.Linear(32,128))
            self.head=nn.Linear(64,1)
        else:
            self.gru=nn.GRU(f,64,batch_first=True); self.head=nn.Linear(64,1)
    def forward(self,x):
        k=self.kind
        if k in ['Logistic','StatsMLP']:
            z=torch.cat([x.mean(1),x.std(1,unbiased=False),x.min(1).values,x.max(1).values],1)
            return self.net(z).squeeze(-1)
        if k=='HistoryMLP': return self.net(x.flatten(1)).squeeze(-1)
        if k=='SingleFrame': return self.net(x[:,-1]).squeeze(-1)
        if k=='Orderless': return self.net(x.mean(1)).squeeze(-1)
        if k=='TCN': return self.head(self.conv(x.transpose(1,2)).mean(2)).squeeze(-1)
        if k in ['M2','M2NoResidual','M2Orderless']:
            a=x[:,:,self.arm]; b=x[:,:,self.body]
            if k=='M2Orderless': a=a.mean(1,keepdim=True)
            _,h=self.a(a); scale,bias=self.b(b.mean(1)).chunk(2,1)
            return self.head(h[-1]*(1+.1*torch.tanh(scale))+.1*bias).squeeze(-1)
        _,h=self.gru(x); return self.head(h[-1]).squeeze(-1)

def normal_response(raw, times, train, names, episodes):
    # 0.4 s state exclusion: target current q+effort; current row q_cmd was
    # already applied before this observation. Current state never enters predictor.
    targets=[j for j,n in enumerate(names) if n.startswith(('q/','effort/'))]
    state=[j for j,n in enumerate(names) if n.startswith(('q/','qdot/'))]
    cmd=[j for j,n in enumerate(names) if n.startswith('q_cmd/')]
    if not targets or not cmd: raise ValueError('normal response feature names not found')
    normal=[i for i in train if episodes[i]['is_control'] and episodes[i]['y']==1]
    if not normal: raise ValueError('no training-only empty controls for normal response')
    def inp(x,t):
        idx=np.maximum(0,np.searchsorted(t,t-.4,side='right')-1)
        return np.concatenate([x[idx][:,state],x[:,cmd]],1)
    valid=[t-t[0]>=.4 for t in times]
    a=np.concatenate([inp(raw[i],times[i])[valid[i]] for i in normal]); b=np.concatenate([raw[i][valid[i]][:,targets] for i in normal])
    mu=a.mean(0); sd=np.maximum(a.std(0),1e-4); z=(a-mu)/sd
    z=np.column_stack([z,np.ones(len(z))]); ridge=np.eye(z.shape[1])*10; ridge[-1,-1]=1e-6
    w=np.linalg.solve(z.T@z+ridge,z.T@b)
    residual=[]
    for i,x in enumerate(raw):
        pred=np.column_stack([(inp(x,times[i])-mu)/sd,np.ones(len(x))])@w
        r=x[:,targets]-pred; r[~valid[i]]=0
        residual.append(np.concatenate([x,r],1).astype('float32'))
    return residual, dict(normal_train_ids=[episodes[i]['trial_id'] for i in normal],
                         target_names=[names[j] for j in targets],state_exclusion_seconds=.4,warmup_residual='zero until 0.4s; warmup excluded from predictor fit',
                         train_mse=float(np.mean((z@w-b)**2))), dict(mu=mu,sd=sd,w=w,targets=targets,state=state,cmd=cmd)

def split_plan(eps):
    # Coarse scene families are stricter than exact parameter groups. No held-out
    # family enters training, even without labels. Positives occur in two families.
    positive=sorted({e['family'] for e in eps if e['y']==1 and not e['is_control']})
    negative=sorted({e['family'] for e in eps if e['y']==0 and not e['is_control']})
    if len(positive)<2: raise ValueError('at least two positive families needed')
    folds=[]
    for k in range(2):
        test=set(positive[k::2]+negative[k::2]); val=set(negative[1-k::2][:1])
        # With only three negative families this leaves very little calibration.
        train=[i for i,e in enumerate(eps) if e['family'] not in test|val]
        va=[i for i,e in enumerate(eps) if e['family'] in val and e['y']>=0]
        te=[i for i,e in enumerate(eps) if e['family'] in test and e['y']>=0 and not e['is_control']]
        folds.append((f'family_fold_{k}',train,va,te))
    return folds

def run(args):
    output=OUT/args.run_name; output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    eps=manifest['episodes']; raw=[]; times=[]; names=None
    for e in eps:
        p=Path(e['npz_path']); p=p if p.is_absolute() else ROOT/p
        d=np.load(p,allow_pickle=False); raw.append(d['seq'].astype('float32')); times.append(d['time'])
        ns=list(d['feature_names'].astype(str))
        if names is not None: assert names==ns
        names=ns
        e['y']=int(e['y']); e['family']=e.get('condition_family_group', e.get('condition_id',e['group_id']))
        if e['is_control'] and e['y']==1: e['family']='EMPTY_CONTROL'
        if e['is_control'] and e['y']==0: e['family']='RIGID_CONTROL'
    if args.limit_methods:
        methods=args.limit_methods.split(',')
    else:
        methods=['Logistic','GradientBoosting','StatsMLP','HistoryMLP','GRU','TCN',
                 'GRUArm','GRUShuffled','Orderless','SingleFrame','GRUPrefix25','GRUPrefix50',
                 'ResidualGRU','M2NoBody','M2NoResidual','M2Orderless','M2']
    folds=split_plan(eps)
    if args.sides:
        folds += [(f'side_{s}',[i for i,e in enumerate(eps) if e['side']==s],[],
                   [i for i,e in enumerate(eps) if e['side']!=s and e['y']>=0 and not e['is_control']]) for s in ['left','right']]
        for s in ['left','right']:
            for label,tr,va,te in split_plan(eps):
                folds.append((f'{label}_{s}',[i for i in tr if eps[i]['side']==s],
                              [i for i in va if eps[i]['side']==s],[i for i in te if eps[i]['side']==s]))
    dump(output/'frozen_config.json',dict(methods=methods,seeds=args.seeds,epochs=args.epochs,
         sequence_length=48,hidden=64,optimizer='AdamW',lr=.003,weight_decay=.01,
         selection='fixed final epoch, no test selection',manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
         folds=[dict(name=n,train=[eps[i]['trial_id'] for i in tr],validation=[eps[i]['trial_id'] for i in va],test=[eps[i]['trial_id'] for i in te]) for n,tr,va,te in folds]))
    # Remove explicit clock from every model; command progression can still
    # encode phase, so shuffled-history ablation cannot prove causality alone.
    keep=[i for i,n in enumerate(names) if n!='elapsed/s']
    names=[names[i] for i in keep]; raw=[x[:,keep] for x in raw]
    arm=[i for i,n in enumerate(names) if any(j in n for j in ['shoulder','elbow','wrist','hand_']) or n=='probe_side/right']
    body=[i for i in range(len(names)) if i not in arm]
    assert arm and body
    predictions=[]; rows=[]; curves=[]; response_audit=[]
    for fold,tr_all,va,te in folds:
        tr=[i for i in tr_all if eps[i]['y']>=0]
        if not tr or not te: continue
        # Group disjointness for config experiments; side experiments explicitly
        # measure cross-side transfer on known configurations, not config OOD.
        if not fold.startswith('side_'):
            assert not {eps[i]['group_id'] for i in tr_all}&{eps[i]['group_id'] for i in te}
        residual,ra,rw=normal_response(raw,times,tr_all,names,eps)
        response_audit.append(dict(fold=fold,**ra)); np.savez_compressed(output/f'{fold}_normal_response.npz',**rw)
        y=np.array([e['y'] for e in eps],dtype='float32')
        fold_methods=methods if fold in ['family_fold_0','family_fold_1'] else ['Logistic','GRU','M2']
        for method in fold_methods:
            uses_res=method in ['ResidualGRU','M2NoBody','M2Orderless','M2']
            data=residual if uses_res else raw
            duration=.5 if method=='GRUPrefix25' else (1.0 if method=='GRUPrefix50' else None)
            x=np.stack([resample(d,times=times[i],duration=duration) for i,d in enumerate(data)])
            m=x[tr_all].mean((0,1)); s=np.maximum(x[tr_all].std((0,1)),1e-3)
            x=np.clip((x-m)/s,-10,10).astype('float32')
            aa=arm+[i for i in range(len(names),x.shape[2]) if any(z in ra['target_names'][i-len(names)] for z in ['shoulder','elbow','wrist'])] if uses_res else arm
            bb=body
            input_indices=list(range(x.shape[2]))
            if method in ['GRUArm','ResidualGRU','M2NoBody']: input_indices=aa
            if method=='Instantaneous':
                # SingleFrame is a last-row summary including accumulated hand
                # progress. This stricter ablation removes that history-derived
                # feature before using only the final instantaneous observation.
                input_indices=[i for i,n in enumerate(names) if 'hand_delta_initial_b' not in n]
            if input_indices!=list(range(x.shape[2])): x=x[:,:,input_indices]
            for seed in args.seeds:
                torch.manual_seed(seed); np.random.seed(seed); xx=x.copy()
                if method=='GRUShuffled':
                    for i in range(len(xx)): xx[i]=xx[i,np.random.default_rng(seed+i*101).permutation(xx.shape[1])]
                start=time.perf_counter(); loss=None
                if method=='GradientBoosting':
                    model=StumpBoost().fit(stat(xx[tr]),y[tr]); p=model.predict(stat(xx)); params=len(model.trees)*4
                    dump(output/f'{fold}_{method}_{seed}.json',dict(bias=model.bias,trees=model.trees,mean=m.tolist(),scale=s.tolist()))
                else:
                    kind='SingleFrame' if method=='Instantaneous' else (method if method in ['Logistic','StatsMLP','HistoryMLP','SingleFrame','Orderless','TCN','M2','M2NoResidual','M2Orderless'] else 'GRU')
                    model=Temporal(kind,xx.shape[2],aa,bb,xx.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                    xt=torch.tensor(xx); yt=torch.tensor(y)
                    for epoch in range(args.epochs):
                        model.train(); opt.zero_grad(); z=model(xt[tr]); loss=nn.functional.binary_cross_entropy_with_logits(z,yt[tr]); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
                        if epoch%10==0 or epoch==args.epochs-1:
                            model.eval()
                            with torch.no_grad(): vl=float(nn.functional.binary_cross_entropy_with_logits(model(xt[va]),yt[va])) if va else None
                            curves.append(dict(fold=fold,method=method,seed=seed,epoch=epoch,train_loss=float(loss),validation_loss=vl))
                    model.eval()
                    with torch.no_grad(): p=torch.sigmoid(model(xt)).numpy()
                    params=sum(v.numel() for v in model.parameters())
                    torch.save(dict(state_dict=model.state_dict(),kind=kind,feature_names=names,arm=aa,body=bb,input_indices=input_indices,mean=m,scale=s,seed=seed,train_ids=[eps[i]['trial_id'] for i in tr]),output/f'{fold}_{method}_{seed}.pt')
                train_seconds=time.perf_counter()-start
                inferstart=time.perf_counter()
                if method=='GradientBoosting': model.predict(stat(xx[te[:1]]))
                else:
                    with torch.no_grad(): model(torch.tensor(xx[te[:1]]))
                infer_ms=(time.perf_counter()-inferstart)*1000
                # Threshold frozen without inspecting test. A 0 observed FP
                # validation threshold is exploratory; with small N no risk bound.
                negval=[i for i in va if y[i]==0]
                high=max(.8,float(max(p[negval]))+1e-6) if negval else 1.000001
                for mode,lo,hi in [('forced',.5,.5),('selective',.2,.8),('validation_zero_fp',.2,high)]:
                    rows.append(dict(fold=fold,method=method,seed=seed,mode=mode,train_n=len(tr),train_positive_n=int(y[tr].sum()),train_control_positive=sum(eps[i]['is_control'] and y[i]==1 for i in tr),train_control_negative=sum(eps[i]['is_control'] and y[i]==0 for i in tr),train_core_positive=sum(not eps[i]['is_control'] and y[i]==1 for i in tr),train_core_negative=sum(not eps[i]['is_control'] and y[i]==0 for i in tr),validation_n=len(va),params=params,train_seconds=train_seconds,inference_ms=infer_ms,low=lo,high=hi,**metrics(y[te],p[te],lo,hi)))
                for i in te:
                    predictions.append(dict(fold=fold,method=method,seed=seed,trial_id=eps[i]['trial_id'],y=int(y[i]),p=float(p[i]),side=eps[i]['side'],family=eps[i]['family'],group_id=eps[i]['group_id']))
                print(json.dumps(dict(fold=fold,method=method,seed=seed,train_n=len(tr),test_n=len(te),**metrics(y[te],p[te]))),flush=True)
    csvwrite(output/'metrics.csv',rows); csvwrite(output/'predictions.csv',predictions); csvwrite(output/'training_curves.csv',curves)
    dump(output/'response_audit.json',response_audit)
    if args.run_name=='phase_a':
        csvwrite(REPORT/'LEARNING_MODEL_COMPARISON.csv',[r for r in rows if r['method'] in ['Logistic','GradientBoosting','StatsMLP','HistoryMLP','GRU','TCN','ResidualGRU','M2']])
        csvwrite(REPORT/'ABLATION_RESULTS.csv',[r for r in rows if r['method'] not in ['Logistic','GradientBoosting','StatsMLP','HistoryMLP','GRU','TCN','ResidualGRU','M2']])
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--manifest',default=str(OUT/'dataset/manifest.json')); a.add_argument('--run-name',default='phase_a'); a.add_argument('--epochs',type=int,default=60); a.add_argument('--seeds',type=int,nargs='+',default=[4070,4071,4072]); a.add_argument('--limit-methods',default=''); a.add_argument('--sides',action='store_true'); run(a.parse_args())
