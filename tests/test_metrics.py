import numpy as np
from sleep_staging.evaluation.metrics import compute_metrics


def test_metrics_manual_example():
    truth=np.array([0,0,1,1]); predictions=np.array([0,1,1,1]); probabilities=np.eye(2)[predictions]
    metrics=compute_metrics(truth,predictions,probabilities,num_classes=2)
    assert metrics["overall"]["accuracy"]==0.75
    np.testing.assert_allclose(metrics["per_class"]["sensitivity"],[0.5,1.0])
    np.testing.assert_allclose(metrics["per_class"]["specificity"],[1.0,0.5])
