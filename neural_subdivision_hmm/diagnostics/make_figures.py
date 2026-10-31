#!/usr/bin/env python3
"""Generate compact SVG diagnostic figures without matplotlib."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; RES=HERE/'results'; RATIOS=[1,2,3,4,6]
def esc(x):return str(x).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def text(x,y,s,size=12,anchor='middle'):return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial" font-size="{size}" fill="#222">{esc(s)}</text>'
def rect(x,y,w,h,c,stroke='white'):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{c}" stroke="{stroke}"/>'
def save(name,body,w,h): (RES/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{body}</svg>')

def confusion():
    names=[('Classical matched','../../results/classical_gaussian_temperature_0_5_predictions.csv')]+[(f'30 seq seed {i}',f'incremental_n30_l10_s{i}_test_predictions.csv') for i in range(3)]
    body=[text(600,28,'Row-normalized test confusion matrices',20)]; cell=39
    for p,(label,path) in enumerate(names):
        if path.startswith('../../'):
            d=pd.read_csv(RES/path)
        else:d=pd.read_csv(RES/path)
        z=d[d.label.isin(RATIOS)]; mat=np.array([[(int(((z.label==r)&(z.pred==s)).sum())) for s in RATIOS] for r in RATIOS]);norm=mat/mat.sum(1,keepdims=True)
        x0=95+(p%2)*580;y0=82+(p//2)*250;body.append(text(x0+110,y0-16,label,15))
        for i,r in enumerate(RATIOS):
            body.append(text(x0-17,y0+i*cell+24,str(r),11))
            body.append(text(x0+i*cell+cell/2,y0-4,str(r),11))
            for j,s in enumerate(RATIOS):
                v=norm[i,j]; color=f'rgb({int(255-150*v)},{int(255-105*v)},{int(255-35*v)})'
                body.append(rect(x0+j*cell,y0+i*cell,cell,cell,color));body.append(text(x0+j*cell+cell/2,y0+i*cell+24,f'{v:.2f}',10))
    save('diagnostic_confusions.svg',''.join(body),1200,590)

def reliability():
    colors=['#d07916','#1463a5','#59a14f','#8064a2']
    files=[('Classical','../../results/classical_gaussian_temperature_0_5_predictions.csv')]+[(f'30 seq seed {i}',f'incremental_n30_l10_s{i}_test_predictions.csv') for i in range(3)]
    body=[text(475,26,'Posterior reliability on exact-label test intervals',19)];x0,y0,w,h=80,55,650,400
    for j in range(6):
        q=j/5;y=y0+h-q*h;x=x0+q*w
        body.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#ddd"/>');body.append(text(x0-7,y+4,f'{q:.1f}',10,'end'))
        body.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0+h}" stroke="#eee"/>');body.append(text(x,y0+h+18,f'{q:.1f}',10))
    body.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0}" stroke="#777" stroke-dasharray="4,4"/>')
    for idx,(label,path) in enumerate(files):
        d=pd.read_csv(RES/path);d=d[d.label.isin(RATIOS)];P=d[[f'post_{r}' for r in RATIOS]].to_numpy();conf=P.max(1);correct=(d.pred.to_numpy()==d.label.to_numpy()).astype(float);bins=np.minimum((conf*10).astype(int),9);pts=[]
        for k in range(10):
            mask=bins==k
            if mask.any():
                xx=x0+conf[mask].mean()*w; yy=y0+h-correct[mask].mean()*h;pts.append((xx,yy));body.append(f'<circle cx="{xx}" cy="{yy}" r="3" fill="{colors[idx]}"/>')
        if pts:body.append(f'<polyline fill="none" stroke="{colors[idx]}" stroke-width="2" points="'+' '.join(f'{x},{y}' for x,y in pts)+'"/>')
        ly=y0+25+idx*22;body.append(f'<line x1="770" y1="{ly}" x2="800" y2="{ly}" stroke="{colors[idx]}" stroke-width="3"/>');body.append(text(810,ly+4,label,12,'start'))
    body.append(text(x0+w/2,y0+h+43,'Mean maximum posterior',13));body.append(text(20,y0+h/2,'Accuracy',13))
    save('diagnostic_calibration.svg',''.join(body),950,530)

def sweep():
    lam=np.array([0,1,2,5,10],dtype=float)
    bacc=np.array([.440699,.466777,.576956,.590863,.649254])
    state_info=np.array([27.467,8.008,2.698,.879,.716])
    body=[text(600,27,'Supervision weight separates acoustic fit from state alignment',19)];x0,y0,w,h=80,55,780,400
    # left axis BAcc, right axis state NLL/label (drawn as normalized 0..1 scale).
    for j in range(6):
        q=j/5;y=y0+h-q*h;x=x0+q*w
        body.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#ddd"/>');body.append(text(x0-8,y+4,f'{q:.1f}',10,'end'))
        body.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0+h}" stroke="#eee"/>')
    xx=[x0+value/10*w for value in lam]
    pts=[(xx[i],y0+h-(bacc[i]-.35)/.45*h) for i in range(5)]
    body.append('<polyline fill="none" stroke="#1463a5" stroke-width="3" points="'+' '.join(f'{x},{y}' for x,y in pts)+'"/>')
    for i,(x,y) in enumerate(pts):body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#1463a5"/>');body.append(text(x,y-10,f'{bacc[i]:.3f}',11))
    spts=[(xx[i],y0+h-min(state_info[i]/30,1)*h) for i in range(5)]
    body.append('<polyline fill="none" stroke="#d07916" stroke-width="3" points="'+' '.join(f'{x},{y}' for x,y in spts)+'"/>')
    for i,(x,y) in enumerate(spts):body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#d07916"/>')
    for i,x in enumerate(xx):body.append(text(x,y0+h+20,str(int(lam[i])),11))
    body.append(text(x0+w/2,y0+h+43,'lambda_state',13));body.append(text(20,y0+h/2,'BAcc',13));body.append(text(920,y0+h/2,'State NLL / label (scaled)',12))
    body.append(f'<line x1="890" y1="{y0+30}" x2="920" y2="{y0+30}" stroke="#1463a5" stroke-width="3"/>');body.append(text(930,y0+34,'Test BAcc',12,'start'))
    body.append(f'<line x1="890" y1="{y0+55}" x2="920" y2="{y0+55}" stroke="#d07916" stroke-width="3"/>');body.append(text(930,y0+59,'Train state NLL',12,'start'))
    save('objective_weight_sweep.svg',''.join(body),1100,540)

def main():
    confusion();reliability();sweep()
if __name__=='__main__':main()
