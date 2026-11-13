import numpy as np
import pytest
from triagerank.benchmark import synthetic_cohort,topk_organ_recall,run

def test_synthetic_data_reproducible():
    assert synthetic_cohort(230,1).equals(synthetic_cohort(230,1))
    assert synthetic_cohort(230,1)['sepsis_label'].nunique()==2

def test_top_two_ranking():
    target=np.array([[1,0,0],[0,0,1],[0,0,0]])
    scores=np.array([[.9,.1,.2],[.8,.7,.01],[.1,.2,.3]])
    assert topk_organ_recall(target,scores,2)==.5
    with pytest.raises(ValueError):topk_organ_recall(target,scores,4)

def test_benchmark_output():
    metrics=run(n=600)
    assert 0<=metrics['sepsis_auroc']<=1
    assert 0<=metrics['organ_top2_hit_rate']<=1
    assert metrics['train_rows']+metrics['test_rows']==600
