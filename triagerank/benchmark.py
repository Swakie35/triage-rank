"""Research benchmark for synthetic triage labels. Not intended for patient care."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES=['age','heart_rate','respiratory_rate','systolic_bp','temperature','creatinine','oxygen_saturation','platelets']
ORGANS=['renal','cardiovascular','respiratory']

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-35,35)))

def synthetic_cohort(n:int=1800,seed:int=21)->pd.DataFrame:
    if n<200:raise ValueError('at least 200 synthetic observations required')
    rng=np.random.default_rng(seed)
    age=np.clip(rng.normal(61,17,n),18,95)
    hr=np.clip(rng.normal(92,21,n),40,185)
    rr=np.clip(rng.normal(20,6,n),8,45)
    sbp=np.clip(rng.normal(113,24,n),55,190)
    temp=np.clip(rng.normal(37.3,.95,n),34,41)
    creat=np.clip(rng.lognormal(.32,.55,n),.3,7)
    ox=np.clip(rng.normal(95,4,n),65,100)
    platelet=np.clip(rng.normal(210,76,n),25,540)
    renal=sigmoid(-1.4+1.7*(creat-1.6)+.018*(age-60))
    cardiac=sigmoid(-1.3-.045*(sbp-110)+.018*(hr-90))
    resp=sigmoid(-1.4+.20*(rr-20)-.18*(ox-94))
    organ_labels=np.stack([rng.binomial(1,renal),rng.binomial(1,cardiac),rng.binomial(1,resp)],axis=1)
    sepsis_p=sigmoid(-1.85+.018*(hr-90)+.055*(rr-20)+.55*(creat-1.3)+.35*(temp-37.3)+.7*organ_labels.sum(axis=1))
    sepsis=rng.binomial(1,sepsis_p)
    frame=pd.DataFrame(dict(zip(FEATURES,[age,hr,rr,sbp,temp,creat,ox,platelet])))
    # Missing values deliberately introduced to exercise the imputer.
    mask=rng.random((n,len(FEATURES)))<.035
    frame=frame.mask(mask)
    frame['sepsis_label']=sepsis
    for i,organ in enumerate(ORGANS):frame[f'{organ}_label']=organ_labels[:,i]
    return frame

def topk_organ_recall(y_true:np.ndarray,scores:np.ndarray,k:int=2)->float:
    if k<=0 or k>scores.shape[1]:raise ValueError('invalid rank width')
    positive=np.flatnonzero(y_true.sum(axis=1)>0)
    if len(positive)==0:return float('nan')
    rankings=np.argsort(scores[positive],axis=1)[:,-k:]
    return float(np.mean([np.any(y_true[j,rankings[i]]>0) for i,j in enumerate(positive)]))

def run(seed:int=21,n:int=1800)->dict:
    df=synthetic_cohort(n,seed)
    y=df['sepsis_label'].to_numpy()
    tr,te=train_test_split(np.arange(len(df)),test_size=.25,random_state=seed,stratify=y)
    def model(): return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=800))
    # Calibration folds see training rows only; test set remains untouched.
    sepsis_model=CalibratedClassifierCV(estimator=model(),method='sigmoid',cv=3)
    sepsis_model.fit(df.loc[tr,FEATURES],y[tr]);prob=sepsis_model.predict_proba(df.loc[te,FEATURES])[:,1]
    organ_scores=[]
    for organ in ORGANS:
        target=df[f'{organ}_label'].to_numpy()
        m=model();m.fit(df.loc[tr,FEATURES],target[tr]);organ_scores.append(m.predict_proba(df.loc[te,FEATURES])[:,1])
    truth=df.loc[te,[f'{organ}_label' for organ in ORGANS]].to_numpy()
    result={'data':'fully synthetic; no MIMIC-IV or patient records',
            'train_rows':len(tr),'test_rows':len(te),
            'sepsis_auroc':round(float(roc_auc_score(y[te],prob)),4),
            'sepsis_auprc':round(float(average_precision_score(y[te],prob)),4),
            'sepsis_brier':round(float(brier_score_loss(y[te],prob)),4),
            'test_prevalence':round(float(np.mean(y[te])),4),
            'organ_top2_hit_rate':round(topk_organ_recall(truth,np.column_stack(organ_scores),k=2),4)}
    return result

def report(path:str|Path):
    r=run();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(r,indent=2)+'\n');return r
