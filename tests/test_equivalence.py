"""Temporary-style equivalence guard against a literal legacy wrapper."""
import torch
from sleep_staging.models.sleep_staging_model import SleepStagingModel


class LegacyNotebookWrapper(SleepStagingModel):
    """The notebook forward expressed independently over the same named layers."""
    def forward(self,x,sequence_start=True):
        batch,windows,time,channels=x.shape
        z,stem=self.compressor(x.view(batch*windows,time,channels).permute(0,2,1))
        feature=self.lstm_head(z,batch,windows,sequence_start=sequence_start)
        feature=torch.nn.functional.layer_norm(self.post_lstm_do(feature),feature.shape[1:])
        return {"logits":self.classifier(feature),"stem_se":stem}


def test_legacy_and_modular_outputs_are_equivalent():
    torch.manual_seed(9); legacy=LegacyNotebookWrapper().eval(); modular=SleepStagingModel().eval(); modular.load_state_dict(legacy.state_dict())
    assert [p.shape for p in legacy.parameters()]==[p.shape for p in modular.parameters()]
    x=torch.randn(2,3,500,1)
    with torch.no_grad(): expected=legacy(x)["logits"]; actual=modular(x)["logits"]
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
