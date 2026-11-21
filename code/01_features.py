"""Parse PhysioNet/CinC 2012 records and build value and measurement-process features.
Raw observations are kept so that test-time measurement shifts can be simulated."""
import os
ROOT=os.environ.get('P12_ROOT','.')
import os, glob, numpy as np, pandas as pd, pickle
RAW=os.path.join(ROOT,'data','raw'); OUT=os.path.join(ROOT,'out')
TS=['Albumin','ALP','ALT','AST','Bilirubin','BUN','Cholesterol','Creatinine','DiasABP','FiO2','GCS','Glucose',
    'HCO3','HCT','HR','K','Lactate','Mg','MAP','MechVent','Na','NIDiasABP','NIMAP','NISysABP','PaCO2','PaO2',
    'pH','Platelets','RespRate','SaO2','SysABP','Temp','TroponinI','TroponinT','Urine','WBC','Weight']
LABS=['Albumin','ALP','ALT','AST','Bilirubin','BUN','Cholesterol','Creatinine','Glucose','HCO3','HCT','K',
      'Lactate','Mg','Na','Platelets','TroponinI','TroponinT','WBC','FiO2','PaCO2','PaO2','pH','SaO2']
# plausibility limits (values outside are set to missing; documented in the paper)
LIM={'Albumin':(0.5,7),'ALP':(1,5000),'ALT':(1,20000),'AST':(1,20000),'Bilirubin':(0,60),'BUN':(1,250),
     'Cholesterol':(20,800),'Creatinine':(0.1,25),'DiasABP':(1,250),'FiO2':(0.21,1.0),'GCS':(3,15),
     'Glucose':(10,2000),'HCO3':(2,60),'HCT':(5,75),'HR':(1,300),'K':(1,12),'Lactate':(0.1,35),'Mg':(0.3,10),
     'MAP':(1,300),'Na':(90,200),'NIDiasABP':(1,250),'NIMAP':(1,300),'NISysABP':(1,300),'PaCO2':(5,200),
     'PaO2':(10,700),'pH':(6.5,8.0),'Platelets':(1,2000),'RespRate':(1,80),'SaO2':(20,100),'SysABP':(1,300),
     'Temp':(25,45),'TroponinI':(0,500),'TroponinT':(0,50),'Urine':(0,5000),'WBC':(0.1,300),'Weight':(20,300)}
def parse(fn):
    d=pd.read_csv(fn); rid=int(d.loc[d.Parameter=='RecordID','Value'].iloc[0])
    t=d.Time.str.split(':',expand=True).astype(int); d['h']=t[0]+t[1]/60
    desc={}
    for p in ['Age','Gender','Height','ICUType','Weight']:
        v=d.loc[(d.Parameter==p)&(d.h==0),'Value']; desc[p]=float(v.iloc[0]) if len(v) else np.nan
    obs=d[d.Parameter.isin(TS)&~((d.Parameter=='Weight')&(d.h==0))][['h','Parameter','Value']].copy()
    # FiO2 recorded as percent in some rows
    m=(obs.Parameter=='FiO2')&(obs.Value>1); obs.loc[m,'Value']=obs.loc[m,'Value']/100
    keep=np.ones(len(obs),bool)
    for p,(lo,hi) in LIM.items():
        mm=(obs.Parameter==p).values; v=obs.Value.values
        keep[mm & ((v<lo)|(v>hi))]=False
    obs=obs[keep | (obs.Parameter=='MechVent').values]
    obs['rid']=rid
    return rid,desc,obs
def load(s):
    R=[parse(f) for f in sorted(glob.glob(f'{RAW}/set-{s}/*.txt'))]
    desc=pd.DataFrame({r:d for r,d,_ in R}).T; obs=pd.concat([o for _,_,o in R]); return desc,obs
if __name__=='__main__':
    data={}
    for s in 'abc':
        desc,obs=load(s)
        oc=pd.read_csv(f'{RAW}/Outcomes-{s}.txt').set_index('RecordID')
        data[s]=dict(desc=desc,obs=obs,out=oc)
        print(s,desc.shape,obs.shape,oc['In-hospital_death'].mean().round(4))
    pickle.dump(data,open(f'{OUT}/raw.pkl','wb'))
