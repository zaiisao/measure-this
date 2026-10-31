#!/usr/bin/env python3
"""Train the classical-covariance neural emission with exact HMM supervision."""
from __future__ import annotations
import argparse,json,random,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from data import RATIOS,load_sequences,sequence_summary
from model import persistence_log_transition
from diagnostics.weighted_train import select,batch_terms,validation_terms
from diagnostics.structured_emission import StructuredNeuralGaussian

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--lambda-state',type=float,default=10)
    ap.add_argument('--mode',choices=['exact','supervised_only','mixed','unsupervised','partial_family'],default='supervised_only')
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--patience',type=int,default=3);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--condition-on-ratio',action='store_true');ap.add_argument('--condition-on-duration',action='store_true')
    a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    splits,norm=load_sequences();train,mode=select(splits['train'],a.mode)
    if a.mode=='exact': val=[q for q in splits['val'] if q.has_audio and q.fully_exact]
    elif a.mode=='unsupervised': val=[q for q in splits['val'] if q.has_audio];mode_val='unlabeled'
    elif a.mode=='partial_family': val=[q for q in splits['val'] if q.has_audio];mode_val='family'
    else: val=[q for q in splits['val'] if q.has_audio and np.isin(q.labels,RATIOS).any()]
    if a.mode not in ('unsupervised','partial_family'): mode_val=mode
    model=StructuredNeuralGaussian.from_classical_fit(m=norm['feature_dim'],condition_on_ratio=a.condition_on_ratio,
                                                       condition_on_duration=a.condition_on_duration,scale_floor=.05)
    # The generic covariance regularizer in batch_terms expects this rank tag for U U^T.
    model.rank=2
    init=torch.full((len(RATIOS),),-np.log(len(RATIOS)));trans=persistence_log_transition(len(RATIOS),.99)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-5)
    results=HERE/'results';results.mkdir(parents=True,exist_ok=True);best=float('inf');best_epoch=0;wait=0;hist=[]
    for epoch in range(1,a.epochs+1):
        model.train();order=np.random.permutation(len(train));tot={'ac':0.,'st':0.,'joint':0.,'n':0,'c':0}
        for ix in range(0,len(order),4):
            batch=[train[j] for j in order[ix:ix+4]];opt.zero_grad(set_to_none=True)
            # The model starts at the known classical covariance; leave its scale
            # departures unregularized in this first controlled comparison.
            loss,terms,reg=batch_terms(model,batch,mode,init,trans,torch.device('cpu'),a.lambda_state,
                                       regularizer_weight=0)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite structured loss in epoch {epoch}, batch {ix}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            n=max(terms.observed_intervals,1);tot['ac']+=float(terms.acoustic_nll_per_observed)*n;tot['joint']+=float(terms.joint_nll_per_observed)*n
            tot['st']+=float(terms.state_nll_sum);tot['n']+=terms.observed_intervals;tot['c']+=terms.informative_intervals
        valterms=validation_terms(model,val,mode_val,init,trans,torch.device('cpu'),a.lambda_state)
        row={'epoch':epoch,'train_acoustic_nll_per_interval':tot['ac']/tot['n'],'train_state_nll_per_constraint':tot['st']/max(tot['c'],1),
             'train_joint_nll_per_interval':tot['joint']/tot['n'],'val_acoustic_nll_per_interval':float(valterms.acoustic_nll_per_observed),
             'val_state_nll_per_constraint':float(valterms.state_nll_per_informative),'val_joint_nll_per_interval':float(valterms.joint_nll_per_observed),
             'val_weighted_objective':float(valterms.loss)};hist.append(row)
        print(f"{a.name} epoch={epoch:02d} acoustic={row['train_acoustic_nll_per_interval']:.3f} state/label={row['train_state_nll_per_constraint']:.3f} val={row['val_weighted_objective']:.3f}",flush=True)
        if row['val_weighted_objective']<best-1e-5:
            best=row['val_weighted_objective'];best_epoch=epoch;wait=0
            torch.save({'model':model.state_dict(),'config':{'structured':True,'mode':a.mode,'lambda_state':a.lambda_state,
                        'seed':a.seed,'condition_on_ratio':a.condition_on_ratio,'condition_on_duration':a.condition_on_duration,
                        'p_stay':.99,'s2':1.,'tm2':1.,'ta2':16.,'width':.03,'scale_floor':.05},'duration_normalization':norm},results/f'{a.name}.pt')
        else:
            wait+=1
            if wait>=a.patience:break
    meta={'name':a.name,'mode':a.mode,'lambda_state':a.lambda_state,'seed':a.seed,'best_epoch':best_epoch,'best_val_objective':best,
          'train_counts':sequence_summary(train),'validation_counts':sequence_summary(val),'history':hist,
            'initial_emission_matches_classical':True,'scale_floor':.05,'checkpoint':str(results/f'{a.name}.pt')}
    (results/f'{a.name}_training.json').write_text(json.dumps(meta,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
if __name__=='__main__':main()
