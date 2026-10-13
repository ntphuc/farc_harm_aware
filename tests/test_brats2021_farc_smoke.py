import numpy as np
import pandas as pd
import torch

from src.brats2021.actions import ACTION_NAMES, all_candidates
from src.brats2021.controller import evaluate, fit, predict
from src.brats2021.error_map import ErrorMap3D, error_targets
from src.brats2021.metrics import case_metrics
from src.brats2021.model import build_brats_segmentor
from src.brats2021.data import labels_to_regions


def test_brats_segmentor_and_error_map_contract():
    x=torch.randn(1,4,32,32,32)
    gt=(torch.rand(1,3,32,32,32)>.85).float()
    model=build_brats_segmentor('unet3d',4,.1)
    z=model(x)
    assert z.shape==(1,3,32,32,32)
    e=ErrorMap3D(4,.1)(x,z)
    assert e.shape==(1,6,32,32,32)
    assert error_targets(z,gt).shape==e.shape


def test_action_bank_and_spacing_hd95():
    rng=np.random.default_rng(1)
    p=rng.random((16,16,16)).astype(np.float32)
    gt=rng.random((16,16,16))>.85
    assert len(all_candidates(p,.5,8))==len(ACTION_NAMES)
    for _,_,candidate in all_candidates(p,.5,8):
        m=case_metrics(candidate,gt,30.,(1.2,1.2,1.0))
        assert np.isfinite(m['dice']) and np.isfinite(m['hd95'])


def test_action_controller_contract():
    rng=np.random.default_rng(0);rows=[]
    for u in range(30):
        reg=['WT','TC','ET'][u%3];gains=[0]+list(rng.normal(.005,.03,4))
        for aid,name in enumerate(ACTION_NAMES):
            rows.append({'case_id':f'c{u}','unit_id':f'u{u}','region':reg,'action_id':aid,'action_name':name,
                'src_fg_fraction':rng.random()*.1,'src_entropy_mean':rng.random(),'src_mc_var_mean':rng.random()*.02,
                'err_fp_mean':rng.random()*.1,'err_fn_mean':rng.random()*.1,
                'region_WT':float(reg=='WT'),'region_TC':float(reg=='TC'),'region_ET':float(reg=='ET'),
                'act_changed_fraction':rng.random()*.02,**{f'act_is_{n}':float(n==name) for n in ACTION_NAMES},
                'base_dice':.8,'candidate_dice':.8+gains[aid],'delta_utility':gains[aid],
                'harmful':int(gains[aid]<-.002),'beneficial':int(gains[aid]>.002),
                'oracle_best_action':ACTION_NAMES[int(np.argmax(gains))],'oracle_best_gain':float(max(gains))})
    df=pd.DataFrame(rows)
    train=df[df.unit_id.isin([f'u{i}' for i in range(20)])]
    test=df[~df.unit_id.isin([f'u{i}' for i in range(20)])]
    b,_=fit(train,42);pv=predict(b,test)
    policy={'min_pred_gain':-.1,'max_pred_harm':1.,'quality_floor':.8,'harm_penalty':.05}
    dec,s=evaluate(pv,policy,.8,.95)
    assert len(dec)==10 and 0<=s['coverage_region']<=1


def test_brats_label_mapping_legacy_0124():
    seg=np.array([[[0,1],[2,4]],[[4,2],[1,0]]],dtype=np.int16)
    m=labels_to_regions(seg,'legacy_0124')
    assert m.shape==(3,*seg.shape)
    assert np.array_equal(m[2],seg==4)
    assert np.array_equal(m[1],np.isin(seg,[1,4]))
    assert np.array_equal(m[0],np.isin(seg,[1,2,4]))


def test_brats_label_mapping_gli_0123():
    seg=np.array([[[0,1],[2,3]],[[3,2],[1,0]]],dtype=np.int16)
    m=labels_to_regions(seg,'gli_0123')
    assert m.shape==(3,*seg.shape)
    assert np.array_equal(m[2],seg==3)
    assert np.array_equal(m[1],np.isin(seg,[1,3]))
    assert np.array_equal(m[0],np.isin(seg,[1,2,3]))


def test_label_scheme_mismatch_is_fatal():
    seg=np.array([[[0,1],[2,3]]],dtype=np.int16)
    try:
        labels_to_regions(seg,'legacy_0124')
    except ValueError as e:
        assert 'incompatible' in str(e)
    else:
        raise AssertionError('Expected label-scheme mismatch to raise')
