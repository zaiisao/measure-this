#!/usr/bin/env python3
"""Validation-selected evaluation for the classical-initialized structured NHMM."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from data import load_sequences
from evaluate import attach_old_family,exact_metrics,family_metrics,score_sequences,work_bootstrap
from structured_emission import StructuredNeuralGaussian

def load_model(path,norm):
    if path is None:return StructuredNeuralGaussian.from_classical_fit(m=norm['feature_dim'])
    ck=torch.load(path,map_location='cpu',weights_only=False);c=ck['config']
    m=StructuredNeuralGaussian.from_classical_fit(m=norm['feature_dim'],s2=c['s2'],tm2=c['tm2'],ta2=c['ta2'],width=c['width'],
        condition_on_ratio=c['condition_on_ratio'],condition_on_duration=c['condition_on_duration'],
        scale_floor=c.get('scale_floor',1e-6))
    m.load_state_dict(ck['model']);return m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='structured_initial');ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--train-p-stay',type=float,default=.99);a=ap.parse_args();torch.set_num_threads(2)
    splits,norm=load_sequences();model=load_model(a.checkpoint,norm);model.eval();val=splits['val'];test=splits['test']
    candidates=[]
    for p in (.95,.99,.999):
        for t in (.25,.5,1.):
            pred,_=score_sequences(model,val,t,p);_,mt=exact_metrics(pred);candidates.append((mt['balanced_accuracy'],-abs(t-1),-abs(p-.99),p,t))
    _,_,_,pbest,tbest=max(candidates)
    raw,raw_nll=score_sequences(model,test,1.,a.train_p_stay);raw=attach_old_family(raw)
    cal,_=score_sequences(model,test,tbest,pbest);cal=attach_old_family(cal)
    _,rm=exact_metrics(raw);exact,cm=exact_metrics(cal);v,_=score_sequences(model,val,tbest,pbest);v=attach_old_family(v);_,fm=family_metrics(cal,v)
    predicted_counts=cal.pred.value_counts().to_dict();adjacent=transitions=0
    for _,g in cal.groupby('performance_id'):
        g=g.sort_values('beat_index');diff=np.diff(g.beat_index.to_numpy());same=diff==1
        adjacent+=int(same.sum());transitions+=int((g.pred.to_numpy()[1:][same]!=g.pred.to_numpy()[:-1][same]).sum())
    res=HERE/'results';res.mkdir(exist_ok=True);cal.to_csv(res/f'{a.name}_all_test_predictions.csv',index=False);exact.to_csv(res/f'{a.name}_test_predictions.csv',index=False)
    out={'name':a.name,'initialized_from_classical':a.checkpoint is None,'train_p_stay':a.train_p_stay,
         'validation_selected_p_stay':pbest,'validation_selected_temperature':tbest,'raw_metrics':rm,
         'calibrated_metrics':cm,'marginal_nll_per_interval':raw_nll,'family_metrics':fm,
         'predicted_state_occupancy_all_intervals':{str(r):int(predicted_counts.get(r,0)) for r in (1,2,3,4,6)},
         'predicted_transition_count_adjacent_beats':transitions,'adjacent_beat_pairs':adjacent,
         'predicted_transitions_per_100_adjacent_beats':100*transitions/max(adjacent,1),
         'work_bootstrap_bacc_ci95':work_bootstrap(exact,lambda d:float(np.mean([(d.loc[d.label==r,'pred']==r).mean() for r in (1,2,3,4,6) if (d.label==r).any()])),n=2000),
         'checkpoint':str(a.checkpoint) if a.checkpoint else None}
    (res/f'{a.name}_evaluation.json').write_text(json.dumps(out,indent=2))
    print(a.name,'raw/cal',rm['balanced_accuracy'],cm['balanced_accuracy'],'NLL',raw_nll,'family AUC',fm['auc'])

if __name__=='__main__':main()
