import os
ROOT=os.environ.get('P12_ROOT','.')
import sys; sys.path.insert(0,os.path.join(os.environ.get('P12_ROOT','.'),'code'))
from figstyle import plt
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(6.2,4.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis('off')
def box(x,y,w,h,t,bold=False,fs=7.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='square,pad=0',lw=0.7,ec='black',fc='white'))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.25)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=0.7,color='black',shrinkA=0,shrinkB=0,mutation_scale=7))
W=88; X0=6
box(X0,88,W,10,'PhysioNet/CinC Challenge 2012: 12,000 adult ICU stays (MIMIC II)\nfirst 48 h of ICU data; outcome: in-hospital death',True)
box(X0,72,40,10,'Set A (development)\nn = 4000; 5-fold CV for tuning')
box(X0+48,72,40,10,'Sets B + C (held-out test)\nn = 8000; never used for fitting')
arr(X0+20,88,X0+20,82); arr(X0+68,88,X0+68,82)
box(X0,50,W,16,'Feature construction per stay\nvalues (180): admission descriptors and first, last, min, max, median per variable\n'
    'indicators (37): ever-measured flags\nprocess (109): indicators + measurement counts + hours to first measurement')
arr(X0+20,72,X0+20,66)
box(X0,28,W,18,'17 model configurations\nimputation: median, chained-equation regression, native (XGBoost)\n'
    'representation: R0 values, R1 + indicators, R2 + process, R3 process only\nlearner: logistic regression, XGBoost')
arr(X0+W/2,50,X0+W/2,46)
box(X0,0,W,24,'Evaluation on sets B + C\n(1) discrimination and calibration, Event 1 score, bootstrap 95% CI\n'
    '(2) measurement-shift stress tests: lab thinning (q = 0.75, 0.5, 0.25),\nno lactate/blood gas, no arterial line\n'
    '(3) SHAP attribution shares   (4) measurement-dropout augmentation')
arr(X0+W/2,28,X0+W/2,24); arr(X0+68,72,X0+68,66)
fig.savefig(os.path.join(ROOT,'fig')+'/Fig1_workflow.png',dpi=600,bbox_inches='tight'); fig.savefig(os.path.join(ROOT,'fig')+'/Fig1_workflow.pdf',bbox_inches='tight')
