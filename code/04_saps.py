"""Paired comparison with SAPS-I on the common subset of test stays with a recorded SAPS-I score."""
import os
ROOT=os.environ.get('P12_ROOT','.')
import pickle, json, numpy as np
from sklearn.metrics import roc_auc_score
OUT=os.path.join(ROOT,'out')
F=pickle.load(open(f'{OUT}/features.pkl','rb')); S=F['S']
import importlib.util,__main__
spec=importlib.util.spec_from_file_location('m',os.path.join(ROOT,'code','02_models.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); __main__.Imputer=m.Imputer
MM=pickle.load(open(f'{OUT}/models.pkl','rb')); yT=MM['yT']
saps=np.r_[S['b']['saps'],S['c']['saps']]; ok=saps>=0
res={'n_common':int(ok.sum()),'deaths_common':int(yT[ok].sum()),'saps_auroc':roc_auc_score(yT[ok],saps[ok])}
rng=np.random.default_rng(2027); y=yT[ok]; sp=saps[ok]; n=len(y)
for key,name in [(('XGB','none','R3'),'xgb_R3'),(('LR','none','R3'),'lr_R3'),(('XGB','native','R2'),'xgb_native_R2')]:
    p=MM['models'][key]['pT'][ok]; a=roc_auc_score(y,p); d=[]; aa=[]
    for _ in range(1000):
        i=rng.integers(0,n,n)
        if y[i].min()==y[i].max(): continue
        aa.append(roc_auc_score(y[i],p[i])); d.append(aa[-1]-roc_auc_score(y[i],sp[i]))
    res[name]=dict(auroc=a,lo=float(np.percentile(aa,2.5)),hi=float(np.percentile(aa,97.5)),d=a-res['saps_auroc'],dlo=float(np.percentile(d,2.5)),dhi=float(np.percentile(d,97.5)))
sa=[]
for _ in range(1000):
    i=rng.integers(0,n,n); sa.append(roc_auc_score(y[i],sp[i]))
res['saps_lo']=float(np.percentile(sa,2.5)); res['saps_hi']=float(np.percentile(sa,97.5))
json.dump(res,open(f'{OUT}/saps_paired.json','w'),indent=1); print(res)
