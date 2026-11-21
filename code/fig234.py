import os
ROOT=os.environ.get('P12_ROOT','.')
import sys; sys.path.insert(0,os.path.join(os.environ.get('P12_ROOT','.'),'code'))
from figstyle import plt, C
import pandas as pd, numpy as np
OUT=os.path.join(ROOT,'out'); FIG=os.path.join(ROOT,'fig')
# ---------- Fig 2: measurement prevalence by outcome (set A)
pv=pd.read_csv(f'{OUT}/measurement_prevalence_A.csv')
pv=pv[pv.meas_all<0.98].sort_values('OR')
fig,ax=plt.subplots(1,2,figsize=(6.2,4.4),gridspec_kw={'width_ratios':[1.25,1]},sharey=True)
y=np.arange(len(pv))
ax[0].hlines(y,pv.meas_surv*100,pv.meas_death*100,color='#BBBBBB',lw=0.8)
ax[0].plot(pv.meas_surv*100,y,'o',ms=3.2,color='#2166AC',label='Survivors')
ax[0].plot(pv.meas_death*100,y,'s',ms=3.2,color='#B2182B',label='Non-survivors')
ax[0].set_yticks(y); ax[0].set_yticklabels(pv['var']); ax[0].set_xlabel('Stays with \u22651 measurement in 48 h (%)')
ax[0].legend(frameon=False,loc='lower left',bbox_to_anchor=(0.0,0.07)); ax[0].set_title('A. Measurement prevalence by outcome',loc='left',fontweight='bold')
ax[1].errorbar(pv.OR,y,xerr=[pv.OR-pv.lo,pv.hi-pv.OR],fmt='o',ms=3,color='black',ecolor='#777777',elinewidth=0.7,capsize=0)
ax[1].axvline(1,ls='--',color='#999999',lw=0.7); ax[1].set_xscale('log'); ax[1].set_xlabel('Odds ratio for death if measured (95% CI)')
from matplotlib.ticker import NullFormatter, FixedLocator, NullLocator
ax[1].xaxis.set_major_locator(FixedLocator([0.25,0.5,1,2,4])); ax[1].set_xticklabels(['0.25','0.5','1','2','4']); ax[1].xaxis.set_minor_locator(NullLocator()); ax[1].xaxis.set_minor_formatter(NullFormatter())
ax[1].set_title('B. Unadjusted association',loc='left',fontweight='bold')
fig.tight_layout(); fig.savefig(f'{FIG}/Fig2_measurement_prevalence.png',dpi=600); fig.savefig(f'{FIG}/Fig2_measurement_prevalence.pdf')
# ---------- Fig 3: robustness to lab thinning
a=pd.read_csv(f'{OUT}/shift_agg.csv')
qmap={'Reference':1.0,'Lab thinning q=0.75':0.75,'Lab thinning q=0.5':0.5,'Lab thinning q=0.25':0.25}
a=a[a.scenario.isin(qmap)].copy(); a['q']=a.scenario.map(qmap)
fig,ax=plt.subplots(2,2,figsize=(6.2,4.6),sharex=True)
panels=[('LR','median','Logistic regression (median imputation)'),('XGB','native','XGBoost (native missing handling)')]
lines=[('R0','none','R0 values only','R0','-','o'),('R1','none','R1 + indicators','R1','-','s'),
       ('R2','none','R2 + process features','R2','-','^'),('R2','MDA','R2 + measurement-dropout augmentation','MDA','--','D')]
for j,(l,i,tt) in enumerate(panels):
    for rep,aug,lab,ck,ls,mk in lines:
        d=a[(a.learner==l)&(a.imputation==i)&(a.rep==rep)&(a.aug==aug)].sort_values('q',ascending=False)
        if len(d)==0: continue
        ax[0,j].plot(d.q,d.auroc,ls=ls,marker=mk,ms=3.5,color=C[ck],label=lab)
        ax[1,j].plot(d.q,d.slope,ls=ls,marker=mk,ms=3.5,color=C[ck],label=lab)
    ax[0,j].set_title(('A. ' if j==0 else 'B. ')+tt,loc='left',fontweight='bold',fontsize=7.8)
    ax[1,j].axhline(1,ls='--',color='#999999',lw=0.7)
    ax[1,j].set_xlabel('Retention probability of laboratory observations (q)')
    ax[0,j].set_ylim(0.815,0.87); ax[1,j].set_ylim(0.8,1.1)
    ax[1,j].set_xticks([1,0.75,0.5,0.25]); ax[1,j].invert_xaxis() if j==0 else None
ax[1,1].invert_xaxis()
ax[0,0].set_ylabel('AUROC (sets B + C)'); ax[1,0].set_ylabel('Calibration slope')
h,lb=ax[0,0].get_legend_handles_labels(); fig.legend(h,lb,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(rect=(0,0.08,1,1)); fig.savefig(f'{FIG}/Fig3_robustness.png',dpi=600); fig.savefig(f'{FIG}/Fig3_robustness.pdf')
# ---------- Fig 4: SHAP
s=pd.read_csv(f'{OUT}/shap_meanabs_xgb_native_R2.csv',index_col=0).iloc[:,0].head(20)[::-1]
pc=set(pd.read_pickle(f'{OUT}/features.pkl')['pcols'])
col=['#E08214' if c in pc else ('#4D4D4D' if c in ('Age','Gender','Height','Weight') or c.startswith('ICUType') else '#2166AC') for c in s.index]
fig,ax=plt.subplots(figsize=(4.6,4.0))
ax.barh(range(len(s)),s.values,color=col,height=0.7); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index)
ax.set_xlabel('Mean |SHAP value| (log-odds), sets B + C')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#2166AC',label='Physiological value'),Patch(color='#E08214',label='Measurement process'),Patch(color='#4D4D4D',label='Admission descriptor')],frameon=False,loc='lower right')
fig.tight_layout(); fig.savefig(f'{FIG}/Fig4_shap.png',dpi=600); fig.savefig(f'{FIG}/Fig4_shap.pdf')
