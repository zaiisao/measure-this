#!/usr/bin/env python3
"""Train existing generative NHMM with decomposed/weighted posterior supervision.

This preserves the normalized emission density and exact HMM forward likelihood.
No classifier logits or CRF scores are introduced.
"""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import RATIOS, constraint_for, load_sequences, sequence_summary
from model import NeuralGaussianEmission, persistence_log_transition
from diagnostics.objective import SequenceTerms, aggregate_terms, sequence_terms

OUT=Path(__file__).resolve().parent/'results'


def select(sequences, mode):
    audio=[q for q in sequences if q.has_audio]
    labeled=[q for q in audio if np.isin(q.labels,RATIOS).any()]
    unlabeled=[q for q in audio if not np.isin(q.labels,RATIOS).any()]
    if mode=='exact': return [q for q in audio if q.fully_exact], 'exact'
    if mode=='supervised_only': return labeled, 'exact'
    if mode=='mixed': return labeled+unlabeled, 'exact'
    if mode=='partial_family': return audio, 'family'
    if mode=='unsupervised': return audio, 'unlabeled'
    raise ValueError(mode)


def sequence_constraints(q, mode, high_confidence_only=False):
    allowed=constraint_for(q,mode)
    if high_confidence_only:
        allowed[~q.high_confidence]=True
    return allowed


def batch_terms(model, seqs, mode, log_init, log_trans, device, lambda_state=1.0,
                regularizer_weight=1e-4, high_confidence_only=False):
    y=torch.from_numpy(np.concatenate([q.y for q in seqs])).to(device)
    d=torch.from_numpy(np.concatenate([q.log_duration for q in seqs])).to(device)
    observed=torch.from_numpy(np.concatenate([q.observed for q in seqs])).to(device)
    emission=model.log_prob(y,d)
    lengths=[q.n for q in seqs]; start=0; terms=[]
    for q,n in zip(seqs,lengths):
        allowed=torch.as_tensor(sequence_constraints(q,mode,high_confidence_only),dtype=torch.bool,device=device)
        terms.append(sequence_terms(log_init,log_trans,emission[start:start+n],allowed,observed[start:start+n]))
        start+=n
    result=aggregate_terms(terms,lambda_state=lambda_state)
    if bool(observed.any()) and regularizer_weight:
        _,var,u=model.distribution_parameters(d[observed])
        reg=var.log().square().mean()
        if model.rank: reg=reg+0.05*u.square().mean()
        loss=result.loss+regularizer_weight*reg
    else:
        reg=result.loss.new_zeros(())
        loss=result.loss
    return loss,result,reg


def gradient_norms(model, acoustic_loss, state_loss):
    params=[p for p in model.parameters() if p.requires_grad]
    def norm(loss):
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return float(torch.sqrt(sum((g.detach().square().sum() for g in grads if g is not None),loss.new_zeros(()))))
    return {'acoustic_grad_l2':norm(acoustic_loss),'state_contribution_grad_l2':norm(state_loss)}


@torch.no_grad()
def validation_terms(model, seqs, mode, log_init, log_trans, device, lambda_state,
                     high_confidence_only=False):
    model.eval(); combined=[]
    for start in range(0,len(seqs),4):
        _,terms,_=batch_terms(model,seqs[start:start+4],mode,log_init,log_trans,device,lambda_state,0,
                              high_confidence_only=high_confidence_only)
        combined.append(SequenceTerms(terms.acoustic_nll_sum,terms.joint_nll_sum,
                                       terms.state_nll_sum,terms.observed_intervals,
                                       terms.informative_intervals))
    return aggregate_terms(combined,lambda_state)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--mode',choices=['exact','supervised_only','mixed','partial_family','unsupervised'],required=True)
    ap.add_argument('--lambda-state',type=float,default=1.0)
    ap.add_argument('--rank',type=int,default=2); ap.add_argument('--hidden',type=int,default=48)
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--patience',type=int,default=3)
    ap.add_argument('--batch-sequences',type=int,default=4); ap.add_argument('--lr',type=float,default=8e-4)
    ap.add_argument('--p-stay',type=float,default=.99); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--gradient-diagnostics',action='store_true')
    ap.add_argument('--high-confidence-only',action='store_true',
                    help='leave low-confidence labels unconstrained; does not remove their audio')
    ap.add_argument('--limit-sequences',type=int,default=0,
                    help='for supervised-only mode, retain the same five fully exact sequences and add a seeded number of other labeled sequences')
    a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
    dev=torch.device('cpu'); splits,norm=load_sequences()
    train,mode=select(splits['train'],a.mode)
    if a.limit_sequences:
        if a.mode!='supervised_only' or a.limit_sequences<5:
            raise ValueError('--limit-sequences requires supervised_only mode and a value >= 5')
        exact=[q for q in train if q.fully_exact]
        extras=[q for q in train if not q.fully_exact]
        rng=np.random.default_rng(20261031)
        extras=[extras[i] for i in rng.permutation(len(extras))]
        train=exact+extras[:a.limit_sequences-len(exact)]
    if a.mode in ('exact','unsupervised','partial_family'):
        val=[q for q in splits['val'] if q.has_audio and (a.mode!='exact' or q.fully_exact)]
    else:
        val=[q for q in splits['val'] if q.has_audio and np.isin(q.labels,RATIOS).any()]
    val_mode='unlabeled' if a.mode=='unsupervised' else mode
    model=NeuralGaussianEmission(ratios=RATIOS,m=norm['feature_dim'],hidden=a.hidden,rank=a.rank).to(dev)
    log_init=torch.full((len(RATIOS),),-np.log(len(RATIOS)),device=dev)
    log_trans=persistence_log_transition(len(RATIOS),a.p_stay,device=dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    OUT.mkdir(parents=True,exist_ok=True); history=[]; best=float('inf'); best_epoch=0; patience=0
    for epoch in range(1,a.epochs+1):
        model.train(); order=np.random.permutation(len(train)); sums={k:0. for k in ['acoustic','state','joint','objective','observed','informative']}; batches=0; grad_diag=None
        for ix in range(0,len(order),a.batch_sequences):
            seqs=[train[j] for j in order[ix:ix+a.batch_sequences]]
            opt.zero_grad(set_to_none=True)
            loss,terms,reg=batch_terms(model,seqs,mode,log_init,log_trans,dev,a.lambda_state,
                                       high_confidence_only=a.high_confidence_only)
            if a.gradient_diagnostics and grad_diag is None and terms.informative_intervals:
                state_contrib=terms.state_nll_sum/max(terms.observed_intervals,1)*a.lambda_state
                grad_diag=gradient_norms(model,terms.acoustic_nll_per_observed,state_contrib)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
            nobs=max(terms.observed_intervals,1)
            sums['acoustic']+=float(terms.acoustic_nll_per_observed)*nobs
            sums['joint']+=float(terms.joint_nll_per_observed)*nobs
            sums['state']+=float(terms.state_nll_sum)
            sums['objective']+=float(terms.loss)*nobs
            sums['observed']+=terms.observed_intervals; sums['informative']+=terms.informative_intervals; batches+=1
        train_row={'epoch':epoch,'acoustic_nll_per_observed':sums['acoustic']/max(sums['observed'],1),
                   'state_nll_per_informative':sums['state']/max(sums['informative'],1),
                   'state_contribution_per_observed':sums['state']/max(sums['observed'],1),
                   'joint_nll_per_observed':sums['joint']/max(sums['observed'],1),
                   'weighted_objective_per_observed':sums['objective']/max(sums['observed'],1),
                   'observed_intervals':int(sums['observed']),'informative_intervals':int(sums['informative']),
                   'gradient_diagnostics_first_batch':grad_diag}
        valterms=validation_terms(model,val,val_mode,log_init,log_trans,dev,a.lambda_state,a.high_confidence_only)
        train_row.update({'val_acoustic_nll_per_observed':float(valterms.acoustic_nll_per_observed),
                          'val_state_nll_per_informative':float(valterms.state_nll_per_informative),
                          'val_joint_nll_per_observed':float(valterms.joint_nll_per_observed),
                          'val_weighted_objective_per_observed':float(valterms.loss)})
        history.append(train_row)
        print(f"{a.name} epoch={epoch:02d} acoustic={train_row['acoustic_nll_per_observed']:.3f} "
              f"state/info={train_row['state_nll_per_informative']:.3f} "
              f"joint={train_row['joint_nll_per_observed']:.3f} objective={train_row['weighted_objective_per_observed']:.3f} "
              f"val={train_row['val_weighted_objective_per_observed']:.3f}",flush=True)
        score=float(valterms.loss)
        if score < best-1e-5:
            best=score; best_epoch=epoch; patience=0
            torch.save({'model':model.state_dict(),'config':{'mode':a.mode,'rank':a.rank,'hidden':a.hidden,
                        'p_stay':a.p_stay,'seed':a.seed,'lambda_state':a.lambda_state},
                        'duration_normalization':norm},OUT/f'{a.name}.pt')
        else:
            patience+=1
            if patience>=a.patience: break
    meta={'name':a.name,'mode':a.mode,'constraint_mode':mode,'val_constraint_mode':val_mode,
          'lambda_state':a.lambda_state,
          'lambda_1_identity':'at lambda=1, acoustic NLL/observed + posterior-state NLL/observed equals original constrained joint NLL/observed',
          'high_confidence_only':a.high_confidence_only,
          'state_normalization':'reported per informative constrained interval, then weighted by n_informative/n_observed in objective',
          'rank':a.rank,'hidden':a.hidden,'p_stay':a.p_stay,'seed':a.seed,'limit_sequences':a.limit_sequences,
          'selected_performance_ids':[q.performance_id for q in train],'epochs_run':len(history),
          'best_epoch':best_epoch,'best_val_weighted_objective':best,'train_counts':sequence_summary(train),
          'validation_counts':sequence_summary(val),'history':history,'checkpoint':str(OUT/f'{a.name}.pt')}
    (OUT.parent/f'{a.name}_diagnostic_training.json').write_text(json.dumps(meta,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

if __name__=='__main__': main()
