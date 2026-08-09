from sleep_staging.models.sleep_staging_model import SleepStagingModel, count_trainable_parameters


def test_reported_model_parameter_count():
    assert count_trainable_parameters(SleepStagingModel(use_se=True)) == 1_515_424
