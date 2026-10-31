#!/usr/bin/env python3
"""Matched evaluation against the saved classical HMM and older family reference."""
from __future__ import annotations
import csv,json,math,sys,gzip
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from data import RATIOS,K,BeatSequence,load_sequences
from model import NeuralGaussianEmission,exclude_missing,persistence_log_transition,forward_backward,viterbi

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; RES=HERE/'results'; CKPT=RES/'checkpoints'
DEVICE=torch.device('cpu')

def auc(y,s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float); n1=int(y.sum()); n0=len(y)-n1
    return float((rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n0 and n1 else float('nan')

def bacc(y,p,classes=RATIOS):
    y=np.asarray(y); p=np.asarray(p)
    vals=[float(np.mean(p[y==c]==c)) for c in classes if np.any(y==c)]
    return float(np.mean(vals)) if vals else float('nan')

def choose_threshold(y,score):
    cands=np.unique(np.r_[0.0,score,1.0])
    return max(cands,key=lambda t:bacc(y,(score>=t).astype(int),(0,1)))

def work_bootstrap(df,metric,n=1000,seed=19):
    rng=np.random.default_rng(seed); works=df.work_id.unique(); values=[]
    groups={w:df[df.work_id==w] for w in works}
    for _ in range(n):
        selected=[groups[w] for w in rng.choice(works,len(works),replace=True)]
        z=pd.concat(selected,ignore_index=True)
        v=metric(z)
        if np.isfinite(v): values.append(v)
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

def transitions(p_stay): return persistence_log_transition(K,p_stay,device=DEVICE)

@torch.no_grad()
def score_sequences(model,seqs,temperature=1.0,p_stay=.99):
    model.eval(); init=torch.full((K,),-math.log(K),device=DEVICE); trans=transitions(p_stay)
    out=[]; nll_sum=0.0; n_obs=0
    for q in seqs:
        y=torch.from_numpy(q.y).to(DEVICE); d=torch.from_numpy(q.log_duration).to(DEVICE)
        observed=torch.from_numpy(q.observed).to(DEVICE)
        ll=model.log_prob(y,d)
        raw=exclude_missing(ll,observed)
        post,logz=forward_backward(init,trans,raw*temperature)
        path=viterbi(init,trans,raw*temperature)
        nll_sum-=float(logz); n_obs+=int(observed.sum())
        posts=post.cpu().numpy(); pred=np.array(RATIOS)[path.cpu().numpy()]
        for i in range(q.n):
            out.append({'performance_id':q.performance_id,'work_id':q.work_id,'stem':q.stem,'beat_index':int(q.beat_index[i]),
                        'label':int(q.labels[i]),'observed':bool(q.observed[i]),'pred':int(pred[i]),'post_max':float(posts[i].max()),
                        'post_1':float(posts[i,0]),'post_2':float(posts[i,1]),'post_3':float(posts[i,2]),'post_4':float(posts[i,3]),'post_6':float(posts[i,4]),
                        'log_duration_z':float(q.log_duration[i])})
    return pd.DataFrame(out),nll_sum/max(n_obs,1)

def attach_old_family(df):
    with gzip.open(ROOT/'beat_aligned_onsets/data/asap_score_reference.csv.gz','rt') as f:
        ref=pd.read_csv(f,usecols=['stem','beat_index','ref_family_ctx4'])
    return df.merge(ref,on=['stem','beat_index'],how='left')

def exact_metrics(df):
    z=df[(df.observed)&df.label.isin(RATIOS)].copy()
    counts={str(r):int((z.label==r).sum()) for r in RATIOS}
    recalls={str(r):float((z.loc[z.label==r,'pred']==r).mean()) for r in RATIOS}
    conf=np.zeros((K,K),dtype=int)
    for i,r in enumerate(RATIOS):
        for j,s in enumerate(RATIOS): conf[i,j]=int(((z.label==r)&(z.pred==s)).sum())
    p=z[[f'post_{r}' for r in RATIOS]].to_numpy(); y=z.label.to_numpy()
    one=np.zeros_like(p)
    for i,r in enumerate(RATIOS): one[:,i]=(y==r)
    brier=float(np.mean(np.sum((p-one)**2,axis=1)))
    confmax=p.max(1); correct=z.pred.to_numpy()==y; bins=np.minimum((confmax*10).astype(int),9)
    ece=float(sum(np.mean(bins==b)*abs(np.mean(confmax[bins==b])-np.mean(correct[bins==b])) for b in range(10) if np.any(bins==b)))
    return z,{'n':len(z),'class_counts':counts,'balanced_accuracy':bacc(y,z.pred.to_numpy()),'accuracy':float(np.mean(y==z.pred)),
              'per_class_recall':recalls,'confusion':conf.tolist(),'ece':ece,'brier':brier,'mean_confidence':float(confmax.mean()),
              'fraction_confidence_ge_0_9':float(np.mean(confmax>=.9))}

def family_metrics(df,val_df=None):
    z=df[df.ref_family_ctx4.isin(['binary','ternary'])].copy()
    z['family_y']=(z.ref_family_ctx4=='ternary').astype(int)
    z['family_score']=np.log(z.post_3+z.post_6+1e-12)-np.log(z.post_2+z.post_4+1e-12)
    if val_df is None: threshold=0.0
    else:
        v=val_df[val_df.ref_family_ctx4.isin(['binary','ternary'])].copy()
        v['family_y']=(v.ref_family_ctx4=='ternary').astype(int)
        v['family_score']=np.log(v.post_3+v.post_6+1e-12)-np.log(v.post_2+v.post_4+1e-12)
        threshold=choose_threshold(v.family_y.to_numpy(),v.family_score.to_numpy())
    z['family_pred']=(z.family_score>=threshold).astype(int)
    return z,{'n':len(z),'binary':int((z.family_y==0).sum()),'ternary':int((z.family_y==1).sum()),
              'auc':auc(z.family_y.to_numpy(),z.family_score.to_numpy()),
              'balanced_accuracy':bacc(z.family_y.to_numpy(),z.family_pred.to_numpy(),(0,1)),
              'accuracy':float(np.mean(z.family_y==z.family_pred)),'threshold_val':float(threshold)}

def model_from_checkpoint(path):
    ck=torch.load(path,map_location=DEVICE,weights_only=False)
    cfg=ck['config']; norm=ck['duration_normalization']
    model=NeuralGaussianEmission(ratios=RATIOS,m=norm['feature_dim'],hidden=cfg['hidden'],rank=cfg['rank']).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    return model,cfg

def classical_direct(test_seqs):
    pred=pd.read_csv(ROOT/'subdivision_hmm/results/experiment/pred_mel50_gauss_hmm.csv')
    by={(r.performance_id,int(r.beat_index)):r for r in pred.itertuples()}
    init=torch.full((K,),-math.log(K),device=DEVICE); trans=transitions(.99); rows=[]; nll=0; nobs=0
    for q in test_seqs:
        em=np.array([[float(getattr(by[(q.performance_id,int(bi))],f'll_{r}')) for r in RATIOS] for bi in q.beat_index])
        e=torch.tensor(em,dtype=torch.float32,device=DEVICE)
        post,z=forward_backward(init,trans,e); path=viterbi(init,trans,e); nll-=float(z); nobs+=q.n
        p=post.numpy(); pr=np.array(RATIOS)[path.numpy()]
        for i in range(q.n): rows.append({'performance_id':q.performance_id,'work_id':q.work_id,'stem':q.stem,'beat_index':int(q.beat_index[i]),'label':int(q.labels[i]),'observed':bool(q.observed[i]),'pred':int(pr[i]),'post_max':float(p[i].max()),**{f'post_{r}':float(p[i,j]) for j,r in enumerate(RATIOS)}})
    return pd.DataFrame(rows),nll/max(nobs,1)

def main():
    torch.set_num_threads(2); splits,norm=load_sequences(); RES.mkdir(exist_ok=True)
    test=splits['test']; val=splits['val']
    records=[]; model_names=['supervised_exact','unsupervised','supervised_only','mixed','partial_family','supervised_diag','capacity16','transition095']
    for name in model_names:
        path=CKPT/f'{name}.pt'
        if not path.exists(): continue
        model,cfg=model_from_checkpoint(path)
        # Raw model posterior and NLL are the proper generative quantities.
        raw_val,_=score_sequences(model,val,1.,cfg['p_stay'])
        raw_val=attach_old_family(raw_val)
        for p in (.95,.99,.999):
            for temp in (.25,.5,1.):
                predv,_=score_sequences(model,val,temp,p)
                predv=attach_old_family(predv)
                ex,met=exact_metrics(predv)
                records.append({'model':name,'p_stay':p,'temperature':temp,'split':'val',**met})
        # Select decoder settings on validation balanced accuracy only.
        candidates=[]
        for p in (.95,.99,.999):
            for temp in (.25,.5,1.):
                predv,_=score_sequences(model,val,temp,p); _,met=exact_metrics(predv)
                candidates.append((met['balanced_accuracy'],-abs(temp-1),-abs(p-.99),p,temp))
        _,_,_,pbest,tbest=max(candidates)
        raw_test,nll=score_sequences(model,test,1.,cfg['p_stay']); raw_test=attach_old_family(raw_test)
        calibrated_test,_=score_sequences(model,test,tbest,pbest); calibrated_test=attach_old_family(calibrated_test)
        z,rawmet=exact_metrics(raw_test); cz,calmet=exact_metrics(calibrated_test)
        valbest,_=score_sequences(model,val,tbest,pbest); valbest=attach_old_family(valbest)
        fam,fammet=family_metrics(calibrated_test,valbest)
        works=z.work_id.nunique()
        ci=work_bootstrap(z,lambda d:bacc(d.label.to_numpy(),d.pred.to_numpy()),n=800)
        family_auc_ci=work_bootstrap(fam,lambda d:auc(d.family_y.to_numpy(),d.family_score.to_numpy()),n=800)
        model.eval()
        with torch.no_grad():
            durations=torch.zeros(1)
            _,var,u=model.distribution_parameters(durations)
            covdiag=var.mean().item(); factor=float(u.square().mean().sqrt()) if cfg['rank'] else 0.
        transitions=sum(int(np.sum(np.diff(g.pred.to_numpy())!=0)) for _,g in cz.groupby('performance_id'))
        transition_rate=100*transitions/max(len(cz),1)
        result={'model':name,'rank':cfg['rank'],'hidden':cfg['hidden'],'p_stay_raw':cfg['p_stay'],'raw_temperature':1.0,
                'test_marginal_nll_per_interval':nll,'raw_metrics':rawmet,'val_selected_p_stay':pbest,'val_selected_temperature':tbest,
                'calibrated_metrics':calmet,'calibrated_work_bootstrap_bacc_95ci':ci,'family_metrics':fammet,'family_auc_work_bootstrap_95ci':family_auc_ci,
                'predicted_transitions_per_100_intervals':transition_rate,'mean_diag_variance':covdiag,'mean_factor_rms':factor,
                'test_works':works,'test_performances':cz.performance_id.nunique()}
        (RES/f'{name}_evaluation.json').write_text(json.dumps(result,indent=2))
        cz.to_csv(RES/f'{name}_test_predictions.csv',index=False)
        records.append({'model':name,'p_stay':pbest,'temperature':tbest,'split':'test','n':calmet['n'],
                        'balanced_accuracy':calmet['balanced_accuracy'],'accuracy':calmet['accuracy'],
                        'ece':calmet['ece'],'brier':calmet['brier'],'marginal_nll':nll,'family_auc':fammet['auc'],
                        'family_balanced_accuracy':fammet['balanced_accuracy'],'test_work_bootstrap_bacc_lo':ci[0],
                        'test_work_bootstrap_bacc_hi':ci[1],'family_auc_work_bootstrap_lo':family_auc_ci[0],
                        'family_auc_work_bootstrap_hi':family_auc_ci[1],'pred_transitions_per_100':transition_rate})
        print(name, 'raw',rawmet['balanced_accuracy'],'cal',calmet['balanced_accuracy'],'NLL',nll,'family AUC',fammet['auc'],flush=True)
    # Direct matched classical HMM recomputed with uniform initial prior, p_stay=.99, kappa=1.
    base,base_nll=classical_direct(test); base=attach_old_family(base); bz,bm=exact_metrics(base)
    bfamily,bfm=family_metrics(base)
    base.to_csv(RES/'classical_gaussian_direct_predictions.csv',index=False)
    records.append({'model':'classical_gaussian_uniform_init_raw','p_stay':.99,'temperature':1.,'split':'test','n':bm['n'],
                    'balanced_accuracy':bm['balanced_accuracy'],'accuracy':bm['accuracy'],'ece':bm['ece'],'brier':bm['brier'],
                    'marginal_nll':base_nll,'family_auc':bfm['auc'],'family_balanced_accuracy':bfm['balanced_accuracy']})
    pd.DataFrame(records).to_csv(RES/'matched_metrics.csv',index=False)
    # Preserve the official baseline separately (train prior + validation-selected temperature).
    official={'source':'mel50','model':'gauss','decoder':'hmm','balanced_accuracy':0.6154344414248734,
              'accuracy':0.6889123692014043,'kappa':0.5,'p_stay':0.99,'test_n':29339,
              'note':'Original matched classical HMM, retained unchanged in subdivision_hmm/results/experiment/results.csv'}
    (RES/'official_classical_baseline.json').write_text(json.dumps(official,indent=2))
    print('classical direct',bm['balanced_accuracy'],'NLL',base_nll,'family AUC',bfm['auc'])

if __name__=='__main__': main()
