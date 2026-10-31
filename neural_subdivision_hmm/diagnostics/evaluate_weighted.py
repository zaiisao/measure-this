#!/usr/bin/env python3
"""Evaluate weighted-supervision checkpoints using validation-only decoder selection."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE.parent.parent))
from data import load_sequences
from evaluate import (attach_old_family, exact_metrics, family_metrics, model_from_checkpoint,
                      score_sequences, work_bootstrap)

RESULTS=HERE/'results'

def evaluate_one(name, checkpoint):
    data,_=load_sequences(); val=data['val']; test=data['test']
    model,cfg=model_from_checkpoint(checkpoint)
    raw_val,_=score_sequences(model,val,1.0,cfg['p_stay']); raw_val=attach_old_family(raw_val)
    candidates=[]
    for p in (.95,.99,.999):
        for temp in (.25,.5,1.0):
            pred,_=score_sequences(model,val,temp,p); _,metrics=exact_metrics(pred)
            candidates.append((metrics['balanced_accuracy'],-abs(temp-1),-abs(p-.99),p,temp))
    _,_,_,pbest,tbest=max(candidates)
    raw,raw_nll=score_sequences(model,test,1.0,cfg['p_stay']); raw=attach_old_family(raw)
    cal,_=score_sequences(model,test,tbest,pbest); cal=attach_old_family(cal)
    _,raw_metrics=exact_metrics(raw); exact,cal_metrics=exact_metrics(cal)
    val_cal,_=score_sequences(model,val,tbest,pbest); val_cal=attach_old_family(val_cal)
    _,fmetrics=family_metrics(cal,val_cal)
    predicted_counts=cal.pred.value_counts().to_dict()
    adjacent=transitions=0
    for _,g in cal.groupby('performance_id'):
        g=g.sort_values('beat_index');diff=np.diff(g.beat_index.to_numpy());same=diff==1
        adjacent+=int(same.sum());transitions+=int((g.pred.to_numpy()[1:][same]!=g.pred.to_numpy()[:-1][same]).sum())
    train_meta=RESULTS/f'{name}_diagnostic_training.json'
    result={'name':name,'lambda_state':cfg.get('lambda_state'),'p_stay_raw':cfg['p_stay'],
            'validation_selected_p_stay':pbest,'validation_selected_temperature':tbest,
            'raw_test_metrics':raw_metrics,'calibrated_test_metrics':cal_metrics,
            'family_metrics':fmetrics,'work_bootstrap_bacc_95ci':work_bootstrap(exact,lambda d: float(np.mean([(d.loc[d.label==r,'pred']==r).mean() for r in (1,2,3,4,6) if (d.label==r).any()])),n=2000),
            'predicted_state_occupancy_all_intervals':{str(r):int(predicted_counts.get(r,0)) for r in (1,2,3,4,6)},
            'predicted_transition_count_adjacent_beats':transitions,'adjacent_beat_pairs':adjacent,
            'predicted_transitions_per_100_adjacent_beats':100*transitions/max(adjacent,1),
            'test_marginal_nll_per_interval':raw_nll,
            'training_metadata':str(train_meta)}
    RESULTS.mkdir(parents=True,exist_ok=True)
    cal.to_csv(RESULTS/f'{name}_all_test_predictions.csv',index=False)
    exact.to_csv(RESULTS/f'{name}_test_predictions.csv',index=False)
    (RESULTS/f'{name}_evaluation.json').write_text(json.dumps(result,indent=2))
    print(name,'lambda',cfg.get('lambda_state'),'raw/cal BAcc',raw_metrics['balanced_accuracy'],cal_metrics['balanced_accuracy'],
          'postNLL',fmetrics.get('auc'),'family AUC',fmetrics['auc'],flush=True)
    return result

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('names',nargs='+'); a=ap.parse_args()
    results=[]
    for name in a.names:
        results.append(evaluate_one(name,RESULTS/f'{name}.pt'))
    (RESULTS/'weighted_comparison.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__': main()
