import logging

from ludwig.api import LudwigModel
from ludwig.datasets import allstate_claims_severity

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

allstate_claims_severity_df = allstate_claims_severity.load()
model.train(
    dataset=allstate_claims_severity_df,
    experiment_name='allstate_claims_severity_transf_sanity_laptop',
    model_name='allstate_claims_severity_transf_sanity_laptop',
    skip_save_processed_input=True
)
