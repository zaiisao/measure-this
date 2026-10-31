#!/usr/bin/env python3
"""Recompute acoustic/state NLL and gradient components for original checkpoints."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from data import RATIOS,load_sequences,constraint_for
from evaluate import model_from_checkpoint
from model import exclude_missing,forward_log_prob,persistence_log_transition
from train import choose_sequences

def run():
    torch.set_num_threads(2);splits,_=load_sequences();k=len(RATIOS)
    init=torch.full((k,),-np.log(k));out={}
    for name,mode in [('supervised_exact','exact'),('supervised_only','supervised_only'),
                      ('mixed','mixed'),('unsupervised','unsupervised')]:
        seqs,constraint_mode=choose_sequences(splits['train'],mode)
        model,cfg=model_from_checkpoint(ROOT/'results/checkpoints'/f'{name}.pt');model.eval()
        trans=persistence_log_transition(k,cfg['p_stay']);ac=joint=0.;nobs=ninfo=0
        gac=[torch.zeros_like(p) for p in model.parameters()];gst=[torch.zeros_like(p) for p in model.parameters()]
        for start in range(0,len(seqs),4):
            batch=seqs[start:start+4];terms=[]
            for q in batch:
                y=torch.from_numpy(q.y);d=torch.from_numpy(q.log_duration);obs=torch.from_numpy(q.observed)
                emission=exclude_missing(model.log_prob(y,d),obs)
                allowed=torch.as_tensor(constraint_for(q,constraint_mode),dtype=torch.bool)
                logp_o=forward_log_prob(init,trans,emission)
                logp_joint=forward_log_prob(init,trans,emission,allowed)
                terms.append((-logp_o,-logp_joint,logp_o-logp_joint,int(obs.sum()),int((~allowed.all(1)).sum())))
            ac_sum=sum(x[0] for x in terms);joint_sum=sum(x[1] for x in terms);state_sum=sum(x[2] for x in terms)
            n=sum(x[3] for x in terms);info=sum(x[4] for x in terms);ac+=float(ac_sum);joint+=float(joint_sum);nobs+=n;ninfo+=info
            if info:
                ga=torch.autograd.grad(ac_sum/max(n,1),tuple(model.parameters()),retain_graph=True,allow_unused=True)
                gs=torch.autograd.grad(state_sum/max(n,1),tuple(model.parameters()),allow_unused=True)
                for j,g in enumerate(ga):
                    if g is not None:gac[j]+=g.detach()
                for j,g in enumerate(gs):
                    if g is not None:gst[j]+=g.detach()
        ac_rate=ac/max(nobs,1);joint_rate=joint/max(nobs,1);state_rate=joint_rate-ac_rate
        grad={}
        if ninfo:
            dot=sum((a*b).sum() for a,b in zip(gac,gst));na=sum(a.square().sum() for a in gac).sqrt();ns=sum(b.square().sum() for b in gst).sqrt()
            grad={'acoustic_gradient_l2_per_observed':float(na),'state_gradient_l2_per_observed':float(ns),
                  'gradient_cosine':float(dot/(na*ns+1e-30))}
        out[name]={'sequences':len(seqs),'observed_intervals':nobs,'informative_intervals':ninfo,
            'acoustic_nll_per_observed':ac_rate,'joint_nll_per_observed':joint_rate,
            'state_nll_per_observed':state_rate,'state_nll_per_informative':state_rate*nobs/max(ninfo,1),**grad}
    return out

if __name__=='__main__':
    result=run();dest=HERE/'existing_checkpoint_decomposition.json';dest.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
