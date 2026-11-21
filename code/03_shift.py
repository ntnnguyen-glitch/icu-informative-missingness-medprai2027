"""Measurement-shift stress tests on pooled sets B+C, measurement-dropout augmentation,
SHAP attribution shares, and descriptive tables."""
import os
ROOT=os.environ.get('P12_ROOT','.')
import pickle, json, numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
import importlib.util
spec=importlib.util.spec_from_file_location('m',os.path.join(ROOT,'code','02_models.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import xgboost as xgb
import __main__
__main__.Imputer=m.Imputer
from sklearn.metrics import roc_auc_score, brier_score_loss
OUT=os.path.join(ROOT,'out'); t0=time.time()
raw=pickle.load(open(f'{OUT}/raw.pkl','rb')); F=pickle.load(open(f'{OUT}/features.pkl','rb')); MM=pickle.load(open(f'{OUT}/models.pkl','rb'))
vcols,mcols,pcols=F['vcols'],F['mcols'],F['pcols']; S=F['S']; models=MM['models']; yT=MM['yT']
descT=pd.concat([raw['b']['desc'],raw['c']['desc']]); obsT=pd.concat([raw['b']['obs'],raw['c']['obs']])
ridsT=S['b']['rids']+S['c']['rids']
ABG=['Lactate','pH','PaCO2','PaO2','SaO2']; ART=['SysABP','DiasABP','MAP']
rng=np.random.default_rng(m.SEED)
SCEN=[('Reference',1.0,0,None)]
for q in [0.75,0.5,0.25]:
    for r in range(3): SCEN.append((f'Lab thinning q={q}',q,r,None))
SCEN+= [('No lactate/ABG',1.0,0,ABG),('No arterial line',1.0,0,ART)]
XS={}
for name,q,r,dv in SCEN:
    o=obsT if dv is None else m.drop(obsT,dv)
    o=m.thin(o,q,np.random.default_rng(m.SEED+int(q*100)+r))
    X,_,_,_=m.build(descT,o,ridsT); XS[(name,r)]=X
    print('built',name,r,round(time.time()-t0))
def evalp(p):
    it,sl=m.calib(yT,p); return dict(auroc=roc_auc_score(yT,p),oe=yT.mean()/p.mean(),slope=sl,brier=brier_score_loss(yT,p),meanp=p.mean())
rows=[]
for key,v in models.items():
    l,i,rep=key
    for (name,r),X in XS.items():
        p=m.predict(v['M'],m.design(X,v['imp'],rep,vcols,mcols,pcols))
        d=evalp(p); d.update(learner=l,imputation=i,rep=rep,scenario=name,rep_id=r,aug='none'); rows.append(d)
    print('eval',key,round(time.time()-t0))
# measurement-dropout augmentation (MDA): train on set A plus thinned copies of set A
XA,yA=S['a']['X'],S['a']['y']; descA=raw['a']['desc']; obsA=raw['a']['obs']; ridsA=S['a']['rids']
copies=[XA]
for j,q in enumerate([0.75,0.5,0.25]):
    Xc,_,_,_=m.build(descA,m.thin(obsA,q,np.random.default_rng(900+j)),ridsA); copies.append(Xc)
XAug=pd.concat(copies); yAug=np.tile(yA,len(copies))
aug_models={}
for key in [('XGB','native','R0'),('XGB','native','R1'),('XGB','native','R2'),('LR','median','R1'),('LR','median','R2'),('XGB','none','R3')]:
    l,i,rep=key; base=models[key]
    imp=m.Imputer(i,vcols).fit(XAug) if rep!='R3' else None
    M=m.fit_learner(l,m.design(XAug,imp,rep,vcols,mcols,pcols),yAug,base['params'])
    aug_models[key]=dict(M=M,imp=imp)
    for (name,r),X in XS.items():
        p=m.predict(M,m.design(X,imp,rep,vcols,mcols,pcols))
        d=evalp(p); d.update(learner=l,imputation=i,rep=rep,scenario=name,rep_id=r,aug='MDA'); rows.append(d)
    print('aug',key,round(time.time()-t0))
sh=pd.DataFrame(rows); sh.to_csv(f'{OUT}/shift_raw.csv',index=False)
agg=sh.groupby(['learner','imputation','rep','aug','scenario'])[['auroc','oe','slope','brier','meanp']].mean().reset_index()
agg.to_csv(f'{OUT}/shift_agg.csv',index=False)
# variability of thinning results is summarised as the range across the three random seeds (not a confidence interval)
# SHAP shares for XGB native R2 on reference test data
v=models[('XGB','native','R2')]; D=m.design(XS[('Reference',0)],v['imp'],'R2',vcols,mcols,pcols)
contrib=v['M'][2].get_booster().predict(xgb.DMatrix(D),pred_contribs=True)[:,:-1]
ma=np.abs(contrib).mean(0); s=pd.Series(ma,index=D.columns).sort_values(ascending=False)
def grp(c):
    if c in pcols: return 'process'
    if c in ('Age','Gender','Height','Weight') or c.startswith('ICUType'): return 'admission descriptors'
    return 'physiological values'
shares=s.groupby([grp(c) for c in s.index]).sum()/s.sum()
s.to_csv(f'{OUT}/shap_meanabs_xgb_native_R2.csv'); shares.to_csv(f'{OUT}/shap_group_shares.csv')
# same for augmented model
va=aug_models[('XGB','native','R2')]; Da=m.design(XS[('Reference',0)],va['imp'],'R2',vcols,mcols,pcols)
ca=va['M'][2].get_booster().predict(xgb.DMatrix(Da),pred_contribs=True)[:,:-1]; sa=pd.Series(np.abs(ca).mean(0),index=Da.columns)
sharesA=sa.groupby([grp(c) for c in sa.index]).sum()/sa.sum(); sharesA.to_csv(f'{OUT}/shap_group_shares_MDA.csv')
print(shares, sharesA)
# measurement prevalence by outcome in set A
prev=[]
for vname in m.MVARS:
    c=XA[f'{vname}_measured'].values
    a=c[yA==1].mean(); b=c[yA==0].mean()
    n11=(c[yA==1]).sum()+.5; n10=(1-c[yA==1]).sum()+.5; n01=(c[yA==0]).sum()+.5; n00=(1-c[yA==0]).sum()+.5
    orr=(n11*n00)/(n10*n01); se=np.sqrt(1/n11+1/n10+1/n01+1/n00)
    prev.append(dict(var=vname,meas_death=a,meas_surv=b,meas_all=c.mean(),OR=orr,lo=np.exp(np.log(orr)-1.96*se),hi=np.exp(np.log(orr)+1.96*se),
                     cnt_death=XA[f'{vname}_count'].values[yA==1].mean(),cnt_surv=XA[f'{vname}_count'].values[yA==0].mean()))
pd.DataFrame(prev).to_csv(f'{OUT}/measurement_prevalence_A.csv',index=False)
# cohort table
rows=[]
for s_ in 'abc':
    X=S[s_]['X']; y=S[s_]['y']; sp=S[s_]['saps']; o=raw[s_]['obs']
    nobs=o.groupby('rid').size()
    r=dict(set=s_.upper(),n=len(y),deaths=int(y.sum()),mort=y.mean(),
           age_med=np.nanmedian(X.Age),age_q1=np.nanpercentile(X.Age,25),age_q3=np.nanpercentile(X.Age,75),
           female=np.nanmean(X.Gender==0),mv=X.MechVent.mean(),
           icu1=X.ICUType_1.mean(),icu2=X.ICUType_2.mean(),icu3=X.ICUType_3.mean(),icu4=X.ICUType_4.mean(),
           saps_med=np.median(sp[sp>=0]),saps_q1=np.percentile(sp[sp>=0],25),saps_q3=np.percentile(sp[sp>=0],75),saps_miss=(sp<0).sum(),
           nobs_med=nobs.median(),nobs_q1=nobs.quantile(.25),nobs_q3=nobs.quantile(.75),
           lact=X.Lactate_measured.mean(),abg=X.pH_measured.mean(),art=X.SysABP_measured.mean(),tni=X.TroponinI_measured.mean(),chol=X.Cholesterol_measured.mean(),
           obs_cells=np.mean([X[f'{v}_measured'].mean() for v in m.MVARS]))
    rows.append(r)
pd.DataFrame(rows).to_csv(f'{OUT}/cohort.csv',index=False)
# augmented models are not pickled (module-scoped class)
print('done',round(time.time()-t0))
