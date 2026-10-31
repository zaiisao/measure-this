#!/usr/bin/env python3
"""Validation-only latent-state permutation analysis for the unsupervised checkpoint."""
import itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from data import RATIOS,load_sequences
from evaluate import exact_metrics,model_from_checkpoint,score_sequences

def score_mapped(df,perm):
    z=df[df.label.isin(RATIOS)].copy()
    idx=np.array([RATIOS.index(int(r)) for r in z.pred])
    z['pred']=np.array(RATIOS)[np.asarray(perm)[idx]]
    return exact_metrics(z)[1]

def main():
    data,_=load_sequences();model,cfg=model_from_checkpoint(ROOT/'results/checkpoints/unsupervised.pt')
    vals=[]
    for p in (.95,.99,.999):
        for temp in (.25,.5,1.):
            v,_=score_sequences(model,data['val'],temp,p)
            for perm in itertools.permutations(range(len(RATIOS))):
                m=score_mapped(v,perm)
                vals.append((m['balanced_accuracy'],m['accuracy'],-abs(temp-1),-abs(p-.99),p,temp,perm,m))
    best=max(vals,key=lambda x:x[:6]);_,_,_,_,pbest,tbest,perm,bval=best
    test,_=score_sequences(model,data['test'],tbest,pbest);_,raw=exact_metrics(test)
    btest=score_mapped(test,perm)
    out={'validation_selected_p_stay':pbest,'validation_selected_temperature':tbest,
         'validation_selected_mapping':{str(RATIOS[i]):RATIOS[perm[i]] for i in range(len(RATIOS))},
         'validation_mapped_metrics':bval,'test_raw_identity_metrics':raw,'test_mapped_metrics':btest,
         'warning':'Permutation was selected only on validation data. Test result is exploratory and does not train or alter the normalized generative model.'}
    path=HERE/'results/unsupervised_state_permutation.json';path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
