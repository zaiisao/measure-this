#!/usr/bin/env python3
"""Train a normalized Gaussian NHMM with exact differentiable HMM marginalization."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import numpy as np
import torch

from data import RATIOS, BeatSequence, constraint_for, load_sequences, sequence_summary
from model import NeuralGaussianEmission, exclude_missing, forward_log_prob, persistence_log_transition

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'/'checkpoints'


def choose_sequences(all_sequences, mode):
    audio=[q for q in all_sequences if q.has_audio]
    labeled=[q for q in audio if np.isin(q.labels,RATIOS).any()]
    unlabeled=[q for q in audio if not np.isin(q.labels,RATIOS).any()]
    if mode=='exact': return [q for q in audio if q.fully_exact], 'exact'
    if mode=='unsupervised': return audio, 'unlabeled'
    if mode=='supervised_only': return labeled, 'exact'
    if mode=='mixed': return labeled+unlabeled, 'exact'
    if mode=='partial_family': return audio, 'family'
    raise ValueError(mode)


def tensors_for(seqs, device):
    ys=torch.from_numpy(np.concatenate([q.y for q in seqs])).to(device)
    ds=torch.from_numpy(np.concatenate([q.log_duration for q in seqs])).to(device)
    observed=torch.from_numpy(np.concatenate([q.observed for q in seqs])).to(device)
    lengths=[q.n for q in seqs]
    return ys,ds,observed,lengths


def nll_batch(model, seqs, constraint_mode, log_init, log_trans, device, reg_weight=1e-4):
    ys,ds,obs,lengths=tensors_for(seqs,device)
    lp=model.log_prob(ys,ds)
    lp=exclude_missing(lp,obs)
    losses=[]; start=0
    for q,n in zip(seqs,lengths):
        allowed_np=constraint_for(q,constraint_mode)
        allowed=torch.as_tensor(allowed_np,dtype=torch.bool,device=device)
        losses.append(-forward_log_prob(log_init,log_trans,lp[start:start+n],allowed))
        start+=n
    total=torch.stack(losses).sum()/max(int(obs.sum()),1)
    # Weak parameter regularization discourages covariance inflation/collapse without
    # replacing or renormalizing the Gaussian density.
    if bool(obs.any()):
        _,var,u=model.distribution_parameters(ds[obs])
        reg=var.log().square().mean()
        if model.rank: reg=reg+0.05*u.square().mean()
    else:
        reg=total.new_zeros(())
    return total+reg_weight*reg, total


@torch.no_grad()
def evaluate_nll(model, seqs, constraint_mode, log_init, log_trans, device):
    if not seqs: return float('nan')
    total_nll=0.0; observed=0
    for i in range(0,len(seqs),4):
        batch=seqs[i:i+4]; loss,nll=nll_batch(model,batch,constraint_mode,log_init,log_trans,device,reg_weight=0)
        _,_,o,_=tensors_for(batch,device); count=int(o.sum()); total_nll+=float(nll)*count; observed+=count
    return total_nll/max(observed,1)


def split_for_mode(seqs, mode):
    chosen,constraint_mode=choose_sequences(seqs,mode)
    if mode=='exact':
        val=[q for q in seqs if q.fully_exact and q.has_audio]
        val_constraint='exact'
    elif mode=='unsupervised':
        val=[q for q in seqs if q.has_audio]; val_constraint='unlabeled'
    elif mode in ('supervised_only','mixed'):
        val=[q for q in seqs if q.has_audio and np.isin(q.labels,RATIOS).any()]
        val_constraint='exact'
    elif mode=='partial_family':
        val=[q for q in seqs if q.has_audio]; val_constraint='family'
    return chosen,constraint_mode,val,val_constraint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',required=True)
    ap.add_argument('--mode',choices=['exact','unsupervised','supervised_only','mixed','partial_family'],required=True)
    ap.add_argument('--rank',type=int,default=2); ap.add_argument('--hidden',type=int,default=48)
    ap.add_argument('--epochs',type=int,default=14); ap.add_argument('--batch-sequences',type=int,default=4)
    ap.add_argument('--lr',type=float,default=8e-4); ap.add_argument('--p-stay',type=float,default=.99)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--patience',type=int,default=4)
    a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    device=torch.device('cpu')
    splits,normalization=load_sequences()
    train,constraint_mode,val,val_mode=split_for_mode(splits['train'],a.mode)
    model=NeuralGaussianEmission(ratios=RATIOS,m=normalization['feature_dim'],hidden=a.hidden,rank=a.rank).to(device)
    log_init=torch.full((len(RATIOS),),-np.log(len(RATIOS)),device=device)
    log_trans=persistence_log_transition(len(RATIOS),a.p_stay,device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    OUT.mkdir(parents=True,exist_ok=True)
    history=[]; best=float('inf'); best_epoch=0; wait=0
    for epoch in range(1,a.epochs+1):
        order=np.random.permutation(len(train)); model.train(); train_losses=[]
        for st in range(0,len(order),a.batch_sequences):
            batch=[train[j] for j in order[st:st+a.batch_sequences]]
            opt.zero_grad(set_to_none=True)
            loss,nll=nll_batch(model,batch,constraint_mode,log_init,log_trans,device)
            if not torch.isfinite(loss): raise FloatingPointError(f'non-finite training loss: {a.name} epoch {epoch}')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
            train_losses.append(float(nll.detach()))
        model.eval(); val_nll=evaluate_nll(model,val,val_mode,log_init,log_trans,device)
        tr=float(np.mean(train_losses))
        history.append({'epoch':epoch,'train_nll_per_observed_interval':tr,'val_nll_per_observed_interval':val_nll})
        print(f"{a.name} epoch={epoch:02d} train_nll={tr:.4f} val_nll={val_nll:.4f}",flush=True)
        if val_nll < best-1e-5:
            best=val_nll; best_epoch=epoch; wait=0
            torch.save({'model':model.state_dict(),'config':{'mode':a.mode,'rank':a.rank,'hidden':a.hidden,'p_stay':a.p_stay,'seed':a.seed},'duration_normalization':normalization},OUT/f'{a.name}.pt')
        else:
            wait+=1
            if wait>=a.patience: break
    (HERE/'results').mkdir(exist_ok=True)
    meta={'name':a.name,'mode':a.mode,'constraint_mode':constraint_mode,'val_constraint_mode':val_mode,'rank':a.rank,'hidden':a.hidden,
          'p_stay':a.p_stay,'seed':a.seed,'epochs_run':len(history),'best_epoch':best_epoch,'best_val_nll':best,
          'normalization':normalization,'train_counts':sequence_summary(train),'validation_counts':sequence_summary(val),
          'train_labeled_interval_count':int(sum(np.isin(q.labels,RATIOS).sum() for q in train)),
          'history':history,'checkpoint':str(OUT/f'{a.name}.pt')}
    (HERE/'results'/f'{a.name}_training.json').write_text(json.dumps(
        meta, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)))

if __name__=='__main__': main()
