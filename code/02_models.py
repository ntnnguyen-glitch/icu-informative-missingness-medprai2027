"""Factorial comparison of imputation x missingness representation x learner, trained on set A,
tested on sets B and C; measurement-shift stress tests; measurement-dropout augmentation."""
import os
ROOT=os.environ.get('P12_ROOT','.')
import pickle, json, numpy as np, pandas as pd, warnings, time, sys
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import statsmodels.api as sm
warnings.filterwarnings('ignore')
OUT=os.path.join(ROOT,'out'); SEED=2027
TS=['Albumin','ALP','ALT','AST','Bilirubin','BUN','Cholesterol','Creatinine','DiasABP','FiO2','GCS','Glucose',
    'HCO3','HCT','HR','K','Lactate','Mg','MAP','MechVent','Na','NIDiasABP','NIMAP','NISysABP','PaCO2','PaO2',
    'pH','Platelets','RespRate','SaO2','SysABP','Temp','TroponinI','TroponinT','Urine','WBC','Weight']
LABS=['Albumin','ALP','ALT','AST','Bilirubin','BUN','Cholesterol','Creatinine','Glucose','HCO3','HCT','K',
      'Lactate','Mg','Na','Platelets','TroponinI','TroponinT','WBC','FiO2','PaCO2','PaO2','pH','SaO2']
STATV=[v for v in TS if v not in ('MechVent','Urine','Weight')]   # 34 variables with 5 summary statistics
MVARS=[v for v in TS if v!='MechVent']                           # 36 variables with process features

SENT=60.0  # hours-to-first-measurement sentinel for never-measured variables

def build(desc, obs, rids):
    o=obs.sort_values(['rid','h'])
    g=o.groupby(['rid','Parameter'])
    st=g.Value.agg(['first','last','min','max','median','count']); tf=g.h.min()
    F=pd.DataFrame(index=rids)
    F['Age']=desc.loc[rids,'Age'].values
    gd=desc.loc[rids,'Gender'].values.astype(float); gd[(gd!=0)&(gd!=1)]=np.nan; F['Gender']=gd
    h=desc.loc[rids,'Height'].values.astype(float); h[(h<100)|(h>250)]=np.nan; F['Height']=h
    w=desc.loc[rids,'Weight'].values.astype(float); w[(w<20)|(w>300)]=np.nan
    wts=st['first'].xs('Weight',level='Parameter') if 'Weight' in st.index.get_level_values(1) else pd.Series(dtype=float)
    w2=pd.Series(w,index=rids); w2=w2.fillna(wts.reindex(rids)); F['Weight']=w2.values
    for k in [1,2,3,4]: F[f'ICUType_{k}']=(desc.loc[rids,'ICUType'].values==k).astype(float)
    U=st.unstack('Parameter')
    for v in STATV:
        for s in ['first','last','min','max','median']:
            F[f'{v}_{s}']=U[(s,v)].reindex(rids).values if (s,v) in U.columns else np.nan
    us=o[o.Parameter=='Urine'].groupby('rid').Value.sum(); F['Urine_sum']=us.reindex(rids).values
    mv=o[o.Parameter=='MechVent'].groupby('rid').size(); F['MechVent']=(mv.reindex(rids).fillna(0).values>0).astype(float)
    vcols=list(F.columns)
    P=pd.DataFrame(index=rids)
    cnt=U['count'] if 'count' in U.columns.get_level_values(0) else None
    TF=tf.unstack('Parameter')
    for v in MVARS:
        c=cnt[v].reindex(rids).fillna(0).values if v in cnt.columns else np.zeros(len(rids))
        P[f'{v}_measured']=(c>0).astype(float)
    P['Height_measured']=(~np.isnan(F['Height'].values)).astype(float)
    mcols=list(P.columns)
    for v in MVARS:
        c=cnt[v].reindex(rids).fillna(0).values if v in cnt.columns else np.zeros(len(rids))
        P[f'{v}_count']=c
        P[f'{v}_hfirst']=TF[v].reindex(rids).fillna(SENT).values if v in TF.columns else SENT
    pcols=list(P.columns)
    return pd.concat([F,P],axis=1), vcols, mcols, pcols

def thin(obs, q, rng, vars_=LABS):
    if q>=1: return obs
    m=obs.Parameter.isin(vars_).values
    keep=~m | (rng.random(len(obs))<q)
    return obs[keep]
def drop(obs, vars_): return obs[~obs.Parameter.isin(vars_)]

class Imputer:
    def __init__(s,kind,vcols): s.kind=kind; s.vcols=vcols
    def fit(s,X):
        V=X[s.vcols]
        if s.kind=='median': s.m=SimpleImputer(strategy='median').fit(V)
        elif s.kind=='mice':
            s.m=IterativeImputer(estimator=BayesianRidge(),max_iter=10,n_nearest_features=30,initial_strategy='median',
                                 random_state=SEED,skip_complete=True).fit(V)
        return s
    def transform(s,X):
        V=X[s.vcols]
        if s.kind=='native': return V.copy()
        return pd.DataFrame(s.m.transform(V),index=V.index,columns=s.vcols)

def design(X, imp, rep, vcols, mcols, pcols):
    if rep=='R3': return X[pcols].copy()
    V=imp.transform(X)
    if rep=='R0': return V
    if rep=='R1': return pd.concat([V,X[mcols]],axis=1)
    if rep=='R2': return pd.concat([V,X[pcols]],axis=1)

def fit_learner(learner, Xd, y, params):
    if learner=='LR':
        sc=StandardScaler().fit(Xd); m=LogisticRegression(C=params['C'],max_iter=5000).fit(sc.transform(Xd),y)
        return ('LR',sc,m)
    m=xgb.XGBClassifier(n_estimators=params['n'],learning_rate=0.03,max_depth=params['d'],min_child_weight=params['mcw'],
                        subsample=0.8,colsample_bytree=0.5,reg_lambda=1.0,eval_metric='logloss',random_state=SEED,n_jobs=1)
    m.fit(Xd,y); return ('XGB',None,m)
def predict(M, Xd):
    k,sc,m=M
    return m.predict_proba(sc.transform(Xd) if k=='LR' else Xd)[:,1]

GRID={'LR':[{'C':c} for c in [0.003,0.01,0.03,0.1]],
      'XGB':[{'d':d,'mcw':5,'n':n} for d in [2,3] for n in [300,600]]}
IMPCACHE={}
def get_imp(kind,X,vcols,key):
    if (kind,key) not in IMPCACHE: IMPCACHE[(kind,key)]=Imputer(kind,vcols).fit(X)
    return IMPCACHE[(kind,key)]
VCACHE={}
class Cached:
    def __init__(s,imp,tag): s.imp=imp; s.tag=tag
    def transform(s,X):
        k=(id(s.imp),s.tag,len(X))
        if k not in VCACHE: VCACHE[k]=s.imp.transform(X)
        return VCACHE[k]

def cv_tune(learner, imp_kind, rep, XA, yA, vcols, mcols, pcols, groups=None):
    skf=StratifiedKFold(5,shuffle=True,random_state=SEED)
    folds=list(skf.split(XA,yA))
    # build fold design matrices once (imputer refit inside each fold)
    D=[]
    for tr,va in folds:
        fi=len(D)
        imp=get_imp(imp_kind,XA.iloc[tr],vcols,fi) if rep!='R3' else None
        D.append((design(XA.iloc[tr],Cached(imp,('tr',fi)) if imp else None,rep,vcols,mcols,pcols),
                  design(XA.iloc[va],Cached(imp,('va',fi)) if imp else None,rep,vcols,mcols,pcols),tr,va))
    best=None
    for p in GRID[learner]:
        oof=np.zeros(len(yA))
        for dtr,dva,tr,va in D:
            oof[va]=predict(fit_learner(learner,dtr,yA[tr],p),dva)
        a=roc_auc_score(yA,oof)
        if best is None or a>best[0]: best=(a,p,oof.copy())
    return best

def event1_threshold(y,p):
    ts=np.quantile(p,np.linspace(0.5,0.99,200)); best=(0,0.5)
    for t in ts:
        pr=p>=t; tp=(pr&(y==1)).sum(); se=tp/max(y.sum(),1); ppv=tp/max(pr.sum(),1); s=min(se,ppv)
        if s>best[0]: best=(s,t)
    return best[1]
def calib(y,p):
    p=np.clip(p,1e-6,1-1e-6); lp=np.log(p/(1-p))
    try:
        sl=sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial()).fit().params[1]
        it=sm.GLM(y,np.ones((len(y),1)),family=sm.families.Binomial(),offset=lp).fit().params[0]
    except Exception: sl,it=np.nan,np.nan
    return it,sl
def metrics(y,p,thr):
    it,sl=calib(y,p); pr=p>=thr; tp=(pr&(y==1)).sum()
    return dict(auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),brier=brier_score_loss(y,p),
                cal_int=it,cal_slope=sl,oe=y.mean()/p.mean(),event1=min(tp/y.sum(),tp/max(pr.sum(),1)))
def boot_ci(y,p,ref=None,B=1000,seed=SEED):
    rng=np.random.default_rng(seed); n=len(y); a=[];d=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if y[i].min()==y[i].max(): continue
        a.append(roc_auc_score(y[i],p[i]))
        if ref is not None: d.append(a[-1]-roc_auc_score(y[i],ref[i]))
    r=dict(lo=np.percentile(a,2.5),hi=np.percentile(a,97.5))
    if ref is not None: r.update(dlo=np.percentile(d,2.5),dhi=np.percentile(d,97.5),p_le0=np.mean(np.array(d)<=0))
    return r

if __name__=='__main__':
    t0=time.time()
    data=pickle.load(open(f'{OUT}/raw.pkl','rb'))
    S={}
    for s in 'abc':
        rids=list(data[s]['desc'].index)
        X,vcols,mcols,pcols=build(data[s]['desc'],data[s]['obs'],rids)
        y=data[s]['out'].loc[rids,'In-hospital_death'].values.astype(int)
        saps=data[s]['out'].loc[rids,'SAPS-I'].values.astype(float)
        S[s]=dict(X=X,y=y,saps=saps,rids=rids)
    pickle.dump(dict(S=S,vcols=vcols,mcols=mcols,pcols=pcols),open(f'{OUT}/features.pkl','wb'))
    print('features',S['a']['X'].shape,len(vcols),len(mcols),len(pcols),round(time.time()-t0))
    XA,yA=S['a']['X'],S['a']['y']
    XT=pd.concat([S['b']['X'],S['c']['X']]); yT=np.r_[S['b']['y'],S['c']['y']]; setT=np.r_[['B']*4000,['C']*4000]
    CONFIGS=[(l,i,r) for l in ['LR','XGB'] for i in ['median','mice'] for r in ['R0','R1','R2']]
    CONFIGS+=[('XGB','native',r) for r in ['R0','R1','R2']]+[('LR','none','R3'),('XGB','none','R3')]
    models={}; rows=[]
    for (l,i,r) in CONFIGS:
        a,p,oof=cv_tune(l,i,r,XA,yA,vcols,mcols,pcols)
        imp=get_imp(i,XA,vcols,'full') if r!='R3' else None
        M=fit_learner(l,design(XA,Cached(imp,'A') if imp else None,r,vcols,mcols,pcols),yA,p)
        thr=event1_threshold(yA,oof)
        pT=predict(M,design(XT,Cached(imp,'T') if imp else None,r,vcols,mcols,pcols))
        models[(l,i,r)]=dict(M=M,imp=imp,params=p,thr=thr,cv_auc=a,pT=pT)
        row=dict(learner=l,imputation=i,rep=r,cv_auroc_A=a,params=json.dumps(p),thr=thr)
        for nm,mask in [('BC',np.ones(len(yT),bool)),('B',setT=='B'),('C',setT=='C')]:
            for k,v in metrics(yT[mask],pT[mask],thr).items(): row[f'{nm}_{k}']=v
        rows.append(row); print(l,i,r,round(a,4),round(row['BC_auroc'],4),p,round(time.time()-t0))
    res=pd.DataFrame(rows)
    # bootstrap CIs and paired differences versus R0 of the same learner/imputation
    for k,row in res.iterrows():
        key=(row.learner,row.imputation,row.rep)
        ref=None
        if row.rep in ('R1','R2'): ref=models[(row.learner,row.imputation,'R0')]['pT']
        if row.rep=='R3': ref=None
        ci=boot_ci(yT,models[key]['pT'],ref)
        for kk,vv in ci.items(): res.loc[k,f'BC_auroc_{kk}']=vv
    ok=S['b']['saps']>=0; ok2=S['c']['saps']>=0
    sapsT=np.r_[S['b']['saps'],S['c']['saps']]; okT=sapsT>=0
    res.attrs['saps']=roc_auc_score(yT[okT],sapsT[okT])
    print('SAPS-I AUROC BC',res.attrs['saps'],okT.sum())
    res.to_csv(f'{OUT}/main_results.csv',index=False)
    json.dump(dict(saps_auroc_BC=res.attrs['saps'],saps_n=int(okT.sum())),open(f'{OUT}/saps.json','w'))
    pickle.dump(dict(models={k:{kk:vv for kk,vv in v.items()} for k,v in models.items()},yT=yT,setT=setT),open(f'{OUT}/models.pkl','wb'))
    print('done',round(time.time()-t0))
