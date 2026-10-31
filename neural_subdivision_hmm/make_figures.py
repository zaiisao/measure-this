#!/usr/bin/env python3
"""Write dependency-free SVG figures and secondary work/subgroup summaries."""
from __future__ import annotations
import gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
RES=HERE/'results'
RATIOS=[1,2,3,4,6]
import sys
sys.path.insert(0,str(HERE))
from model import forward_backward,persistence_log_transition

def bacc(d):
    z=d[d.label.isin(RATIOS)]
    vals=[float((z.loc[z.label==r,'pred']==r).mean()) for r in RATIOS if (z.label==r).any()]
    return float(np.mean(vals)) if vals else float('nan')

def calibrate_classical(d,p_stay=.99,temperature=.5):
    source=pd.read_csv(HERE.parent/'subdivision_hmm/results/experiment/pred_mel50_gauss_hmm.csv',
                       usecols=['performance_id','beat_index']+[f'll_{r}' for r in RATIOS])
    out=d.merge(source,on=['performance_id','beat_index'],how='left'); init=torch.full((5,),-np.log(5.0)); trans=persistence_log_transition(5,p_stay)
    for _,idx in out.groupby('performance_id',sort=False).groups.items():
        ids=np.array(list(idx)); ids=ids[np.argsort(out.loc[ids,'beat_index'].to_numpy())]
        breaks=np.flatnonzero(np.diff(out.loc[ids,'beat_index'].to_numpy())!=1)+1
        for segment in np.split(ids,breaks):
            z=out.loc[segment]; emit=torch.tensor(z[[f'll_{r}' for r in RATIOS]].to_numpy(),dtype=torch.float32)*temperature
            post,_=forward_backward(init,trans,emit); path=post.argmax(1).numpy()
            for j,r in enumerate(RATIOS): out.loc[segment,f'post_{r}']=post[:,j].numpy()
            out.loc[segment,'pred']=np.array(RATIOS)[path]
            out.loc[segment,'post_max']=post.max(1).values.numpy()
    return out

def confidence_metrics(d):
    z=d[d.label.isin(RATIOS)]; p=z[[f'post_{r}' for r in RATIOS]].to_numpy(); y=z.label.to_numpy(); pred=z.pred.to_numpy()
    one=np.zeros_like(p)
    for j,r in enumerate(RATIOS): one[:,j]=(y==r)
    conf=p.max(1); bins=np.minimum((conf*10).astype(int),9); correct=(pred==y)
    ece=sum(np.mean(bins==k)*abs(np.mean(conf[bins==k])-np.mean(correct[bins==k])) for k in range(10) if np.any(bins==k))
    return {'n':len(z),'accuracy':float(correct.mean()),'balanced_accuracy':bacc(z),'ece':float(ece),'brier':float(np.mean(np.sum((p-one)**2,axis=1)))}

def paired_bootstrap(a,b,n=5000,seed=199):
    # Resample works jointly and compute balanced accuracy from summed class counts.
    works=sorted(set(a.work_id)&set(b.work_id)); rng=np.random.default_rng(seed)
    def counts(d):
        z=d[d.label.isin(RATIOS)]
        return {w:np.array([[(int(((z.work_id==w)&(z.label==r)&(z.pred==r)).sum())),
                             int(((z.work_id==w)&(z.label==r)).sum())] for r in RATIOS]) for w in works}
    ca,cb=counts(a),counts(b); vals=[]
    for _ in range(n):
        selected=rng.choice(works,len(works),replace=True)
        aa=sum((ca[w] for w in selected),np.zeros((len(RATIOS),2),dtype=int))
        bb=sum((cb[w] for w in selected),np.zeros((len(RATIOS),2),dtype=int))
        use=(aa[:,1]>0)&(bb[:,1]>0)
        vals.append(float(np.mean(aa[use,0]/aa[use,1]-bb[use,0]/bb[use,1])))
    return {'n_work_bootstraps':n,'delta_bacc_nhmm_minus_classical':float(bacc(a)-bacc(b)),
            'paired_delta_ci95':[float(x) for x in np.quantile(vals,[.025,.975])],
            'bootstrap_fraction_delta_le_zero':float(np.mean(np.array(vals)<=0))}

def esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def write_svg(name,body,width=900,height=620):
    (RES/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><style>text{{font-family:Arial,sans-serif;fill:#222}} .small{{font-size:12px}} .label{{font-size:14px}} .title{{font-size:20px;font-weight:bold}}</style>{body}</svg>')
def rect(x,y,w,h,fill,stroke='none'): return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}"/>'
def txt(x,y,s,size=13,anchor='middle',fill='#222'): return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{fill}">{esc(s)}</text>'

def confusion_figure(a,b):
    W,H=900,460; body=[txt(450,30,'Held-out confusion matrices (validation-selected decode)',20)]
    for panel,(name,d) in enumerate([('Neural Gaussian HMM',a),('Classical Gaussian HMM',b)]):
        z=d[d.label.isin(RATIOS)]; mat=np.array([[(int(((z.label==r)&(z.pred==s)).sum())) for s in RATIOS] for r in RATIOS]); norm=mat/mat.sum(1,keepdims=True)
        x0=95+panel*435; y0=92; cell=56
        body.append(txt(x0+140,y0-30,name,16))
        for i,r in enumerate(RATIOS):
            body.append(txt(x0-20,y0+i*cell+34,str(r),13))
            body.append(txt(x0+i*cell+cell/2,y0-8,str(r),13))
            for j,s in enumerate(RATIOS):
                v=norm[i,j]; color=f'rgb({int(255-150*v)},{int(255-105*v)},{int(255-35*v)})'
                body.append(rect(x0+j*cell,y0+i*cell,cell,cell,color,'white'))
                body.append(txt(x0+j*cell+cell/2,y0+i*cell+34,f'{v:.2f}',12))
        body.append(txt(x0+140,y0+cell*5+30,'Predicted ratio',13))
        body.append(txt(x0-55,y0+cell*2.5,'True ratio',13))
    write_svg('confusion_matrices.svg',''.join(body),W,H)

def reliability_figure(a,b):
    body=[txt(450,28,'Confidence reliability on exact-label test intervals',20)]
    x0,y0,w,h=95,65,650,430
    for i in range(6):
        y=y0+h-i*h/5; body.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#ddd"/>'); body.append(txt(x0-12,y+4,f'{i/5:.1f}',12,'end'))
        x=x0+i*w/5; body.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0+h}" stroke="#eee"/>'); body.append(txt(x,y0+h+20,f'{i/5:.1f}',12))
    body.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0}" stroke="#777" stroke-dasharray="5,5"/>')
    colors=[('#1463a5','Neural NHMM'),('#d07916','Classical HMM')]
    for (color,label),d in zip(colors,[a,b]):
        z=d[d.label.isin(RATIOS)].copy(); ps=z[[f'post_{r}' for r in RATIOS]].to_numpy(); conf=ps.max(1); correct=(z.pred.to_numpy()==z.label.to_numpy()).astype(float); bins=np.minimum((conf*10).astype(int),9); pts=[]
        for k in range(10):
            q=bins==k
            if q.any():
                cx=float(conf[q].mean()); cy=float(correct[q].mean()); px=x0+cx*w; py=y0+h-cy*h; pts.append((px,py)); body.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')
        if pts: body.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(f'{x},{y}' for x,y in pts)+'"/>')
    body.append(txt(x0+w/2,y0+h+48,'Mean maximum posterior',14)); body.append(txt(25,y0+h/2,'Accuracy',14))
    body.append(f'<line x1="{x0+w+30}" y1="{y0+30}" x2="{x0+w+60}" y2="{y0+30}" stroke="#1463a5" stroke-width="3"/>'); body.append(txt(x0+w+68,y0+34,'Neural NHMM',12,'start'))
    body.append(f'<line x1="{x0+w+30}" y1="{y0+55}" x2="{x0+w+60}" y2="{y0+55}" stroke="#d07916" stroke-width="3"/>'); body.append(txt(x0+w+68,y0+59,'Classical HMM',12,'start'))
    write_svg('calibration_reliability.svg',''.join(body),900,570)

def learning_curve_figure():
    names=['supervised_exact','supervised_only','unsupervised','mixed','partial_family']
    colors=['#1463a5','#59a14f','#e15759','#f28e2b','#8064a2']; body=[txt(450,28,'Training and validation negative log likelihood',20)]
    x0,y0,w,h=80,55,730,440; ymin,ymax=70,240
    for j in range(6):
        val=ymin+(ymax-ymin)*j/5; y=y0+h-j*h/5; body.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#ddd"/>'); body.append(txt(x0-8,y+4,f'{val:.0f}',12,'end'))
    for i in range(11):
        x=x0+i*w/10; body.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0+h}" stroke="#eee"/>'); body.append(txt(x,y0+h+20,str(i),12))
    for idx,name in enumerate(names):
        f=RES/f'{name}_training.json'
        if not f.exists(): continue
        d=json.loads(f.read_text()); hist=d['history']; color=colors[idx]
        for key,opacity,dash in [('train_nll_per_observed_interval','0.45','4,3'),('val_nll_per_observed_interval','1','none')]:
            points=[]
            for z in hist:
                xx=x0+(z['epoch']-1)*w/10; yy=y0+h-(z[key]-ymin)*h/(ymax-ymin); points.append((xx,yy))
            body.append(f'<polyline fill="none" stroke="{color}" stroke-opacity="{opacity}" stroke-width="2" stroke-dasharray="{dash}" points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in points)+'"/>')
        body.append(f'<line x1="830" y1="{82+idx*27}" x2="855" y2="{82+idx*27}" stroke="{color}" stroke-width="3"/>'); body.append(txt(862,86+idx*27,name,12,'start'))
    body.append(txt(x0+w/2,y0+h+48,'Epoch',14)); body.append(txt(22,y0+h/2,'NLL / observed interval',14))
    body.append(txt(820,58,'Solid: validation',11,'start')); body.append(txt(820,71,'Dashed: train',11,'start'))
    write_svg('learning_curves.svg',''.join(body),1000,560)

def main():
    RES.mkdir(exist_ok=True)
    a=pd.read_csv(RES/'supervised_exact_test_predictions.csv'); b=pd.read_csv(RES/'classical_gaussian_direct_predictions.csv')
    bcal=calibrate_classical(b); bcal.to_csv(RES/'classical_gaussian_temperature_0_5_predictions.csv',index=False)
    pair=paired_bootstrap(a,bcal); (RES/'paired_bootstrap.json').write_text(json.dumps(pair,indent=2))
    (RES/'calibration_metrics.json').write_text(json.dumps({'supervised_exact_val_selected':confidence_metrics(a),
        'classical_uniform_init_p_0_99_temperature_0_5':confidence_metrics(bcal)},indent=2))
    works=[]
    for name,d in [('supervised_exact',a),('classical_gaussian_uniform_init_p_0_99_temp_0_5',bcal)]:
        for work,g in d.groupby('work_id'):
            exact=g[g.label.isin(RATIOS)]; works.append({'model':name,'work_id':work,'intervals':len(exact),'balanced_accuracy':bacc(g),'accuracy':float((exact.label==exact.pred).mean())})
    pd.DataFrame(works).to_csv(RES/'per_work_metrics.csv',index=False)
    with gzip.open(HERE.parent/'beat_aligned_onsets/data/asap_score_reference.csv.gz','rt') as f:
        ref=pd.read_csv(f,usecols=['stem','beat_index','ref_ambiguous','ref_partial_beat'])
    rows=[]
    for name,d in [('supervised_exact',a),('classical_gaussian_uniform_init_p_0_99_temp_0_5',bcal)]:
        z=d.merge(ref,on=['stem','beat_index'],how='left')
        for col in ['ref_ambiguous','ref_partial_beat']:
            for flag in (False,True):
                g=z[(z[col]==flag)&z.label.isin(RATIOS)]
                rows.append({'model':name,'subgroup':col,'value':flag,'n':len(g),'balanced_accuracy':bacc(g),'accuracy':float((g.label==g.pred).mean()) if len(g) else float('nan')})
    pd.DataFrame(rows).to_csv(RES/'score_ambiguity_metrics.csv',index=False)
    # Compare accuracy across quartiles of single-interval acoustic evidence margin.
    sys.path.insert(0,str(HERE)); from data import load_sequences
    from model import NeuralGaussianEmission
    splits,norm=load_sequences(); ck=torch.load(RES/'checkpoints/supervised_exact.pt',map_location='cpu',weights_only=False)
    cfg=ck['config']; net=NeuralGaussianEmission(m=norm['feature_dim'],hidden=cfg['hidden'],rank=cfg['rank']); net.load_state_dict(ck['model']); net.eval()
    margins=[]
    with torch.no_grad():
        for q in splits['test']:
            lp=net.log_prob(torch.from_numpy(q.y),torch.from_numpy(q.log_duration)).numpy()
            sorted_lp=np.sort(lp,axis=1); margins.extend({'performance_id':q.performance_id,'beat_index':int(bi),'nhmm_margin':float(v)} for bi,v in zip(q.beat_index,sorted_lp[:,-1]-sorted_lp[:,-2]))
    margin=pd.DataFrame(margins)
    classical_ll=pd.read_csv(HERE.parent/'subdivision_hmm/results/experiment/pred_mel50_gauss_hmm.csv',usecols=['performance_id','beat_index']+[f'll_{r}' for r in RATIOS])
    classical_ll['classical_margin']=np.sort(classical_ll[[f'll_{r}' for r in RATIOS]].to_numpy(),axis=1)[:,-1]-np.sort(classical_ll[[f'll_{r}' for r in RATIOS]].to_numpy(),axis=1)[:,-2]
    evidence=[]
    for name,d,col in [('supervised_exact',a,'nhmm_margin'),('classical_gaussian_matched',bcal,'classical_margin')]:
        m=margin if col=='nhmm_margin' else classical_ll
        z=d.merge(m[['performance_id','beat_index',col]],on=['performance_id','beat_index'],how='left'); z=z[z.label.isin(RATIOS)].copy()
        z['margin_quartile']=pd.qcut(z[col],4,labels=['Q1 weakest','Q2','Q3','Q4 strongest'],duplicates='drop')
        for q,g in z.groupby('margin_quartile',observed=True): evidence.append({'model':name,'local_margin_quartile':str(q),'n':len(g),'balanced_accuracy':bacc(g),'accuracy':float((g.label==g.pred).mean()),'mean_margin':float(g[col].mean())})
    pd.DataFrame(evidence).to_csv(RES/'local_evidence_metrics.csv',index=False)
    confusion_figure(a,bcal); reliability_figure(a,bcal); learning_curve_figure()
    print(json.dumps(pair,indent=2)); print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(evidence).to_string(index=False))

if __name__=='__main__': main()
