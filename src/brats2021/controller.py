from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .actions import ACTION_NAMES

@dataclass
class Bundle:
    quality_model: object; gain_model: object; harm_model: object; source_columns:list[str]; action_columns:list[str]; metadata:dict

def reg(seed):return Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('reg',HistGradientBoostingRegressor(max_iter=300,learning_rate=.04,max_leaf_nodes=15,l2_regularization=.5,random_state=seed))])
def clf(seed):return Pipeline([('impute',SimpleImputer(strategy='median')),('clf',RandomForestClassifier(n_estimators=500,max_depth=9,min_samples_leaf=3,class_weight='balanced_subsample',random_state=seed,n_jobs=-1))])

def cols(df):
    src=sorted([c for c in df.columns if c.startswith(('src_','err_','region_'))])
    act=src+sorted([c for c in df.columns if c.startswith('act_')])
    return src,act

def fit(df,seed=42):
    src,act=cols(df); by=df.sort_values(['unit_id','action_id']).drop_duplicates('unit_id')
    q=reg(seed);q.fit(by[src],by.base_dice.astype(float))
    nz=df[df.action_id.astype(int)>0].copy();g=reg(seed+1);g.fit(nz[act],nz.delta_utility.astype(float))
    y=nz.harmful.astype(int);h=None;const=None
    if y.nunique()>1:h=clf(seed+2);h.fit(nz[act],y)
    else:const=float(y.iloc[0]) if len(y) else 0.0
    pq=q.predict(by[src]);pg=g.predict(nz[act]);m={'quality_mae':float(mean_absolute_error(by.base_dice,pq)),'gain_mae':float(mean_absolute_error(nz.delta_utility,pg)),'n_units':int(by.unit_id.nunique())}
    if h is not None:m['harm_auroc']=float(roc_auc_score(y,h.predict_proba(nz[act])[:,1]))
    else:m['harm_auroc']=None
    return Bundle(q,g,h,src,act,{'seed':seed,'harm_constant':const}),m

def predict(b,df):
    x=df.copy();x['pred_base_dice']=b.quality_model.predict(x[b.source_columns]);x['pred_delta_utility']=0.;x['pred_harm_prob']=0.;nz=x.action_id.astype(int)>0
    if nz.any():
        x.loc[nz,'pred_delta_utility']=b.gain_model.predict(x.loc[nz,b.action_columns])
        x.loc[nz,'pred_harm_prob']=float(b.metadata.get('harm_constant',0) or 0) if b.harm_model is None else b.harm_model.predict_proba(x.loc[nz,b.action_columns])[:,1]
    return x

def save(b,p):Path(p).parent.mkdir(parents=True,exist_ok=True);joblib.dump(b,p)
def load(p):return joblib.load(p)
def cp_upper(k,n,confidence=.95):
    if n<=0 or k>=n:return 1.0
    return float(beta.ppf(confidence,k+1,n-k))

def select(g,policy):
    g=g.sort_values('action_id');base=g[g.action_id.astype(int)==0].iloc[0];q=float(base.pred_base_dice);nz=g[g.action_id.astype(int)>0].copy();nz['safe_score']=nz.pred_delta_utility.astype(float)-float(policy.get('harm_penalty',.05))*nz.pred_harm_prob.astype(float)
    safe=nz[(nz.pred_delta_utility.astype(float)>=float(policy['min_pred_gain']))&(nz.pred_harm_prob.astype(float)<=float(policy['max_pred_harm']))]
    if len(safe):
        r=safe.sort_values(['safe_score','pred_delta_utility'],ascending=False).iloc[0];return {'decision':ACTION_NAMES[int(r.action_id)],'action_id':int(r.action_id),'human_review':False,'pred_base_dice':q,'pred_gain':float(r.pred_delta_utility),'pred_harm':float(r.pred_harm_prob)}
    if q>=float(policy['quality_floor']):return {'decision':'no_correction','action_id':0,'human_review':False,'pred_base_dice':q,'pred_gain':0.,'pred_harm':0.}
    return {'decision':'human_review','action_id':-1,'human_review':True,'pred_base_dice':q,'pred_gain':0.,'pred_harm':1.}

def evaluate(df,policy,failure_dice=.80,confidence=.95):
    out=[]
    for uid,g in df.groupby('unit_id',sort=False):
        d=select(g,policy);base=g[g.action_id.astype(int)==0].iloc[0]
        if d['human_review']:chosen=base;final=np.nan;harm=np.nan
        else:chosen=g[g.action_id.astype(int)==d['action_id']].iloc[0];final=float(chosen.candidate_dice);harm=int(chosen.harmful)
        out.append({'unit_id':uid,'case_id':base.case_id,'region':base.region,**d,'base_dice':float(base.base_dice),'final_dice':final,'true_harmful':harm,'oracle_best_action':base.oracle_best_action,'oracle_best_gain':float(base.oracle_best_gain)})
    dec=pd.DataFrame(out);auto=dec[~dec.human_review];fails=int((auto.final_dice<float(failure_dice)).sum()) if len(auto) else 0
    # Case-level referral if any region asks for review.
    case_review=dec.groupby('case_id').human_review.max()
    s={'n_units':len(dec),'n_cases':int(dec.case_id.nunique()),'coverage_region':float(len(auto)/max(len(dec),1)),'review_rate_region':float(dec.human_review.mean()),'review_rate_case':float(case_review.mean()),'auto_failure_rate':float(fails/max(len(auto),1)) if len(auto) else 1.0,'auto_failure_upper':cp_upper(fails,len(auto),confidence),'auto_mean_dice':float(auto.final_dice.mean()) if len(auto) else None,'baseline_mean_dice':float(dec.base_dice.mean()),'harmful_action_rate':float(auto.true_harmful.fillna(0).mean()) if len(auto) else 0.0,'correction_rate':float((auto.action_id>0).mean()) if len(auto) else 0.0}
    return dec,s
