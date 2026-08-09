import torch
from sleep_staging.models.sleep_staging_model import SleepStagingModel


def test_forward_shape_and_determinism():
    torch.manual_seed(1); model=SleepStagingModel().eval(); x=torch.randn(2,3,500,1)
    with torch.no_grad(): first=model(x)["logits"]; second=model(x)["logits"]
    assert first.shape==(2,5); torch.testing.assert_close(first,second)
