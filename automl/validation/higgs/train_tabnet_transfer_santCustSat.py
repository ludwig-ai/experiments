import logging

from ludwig.api import LudwigModel
from ludwig.datasets import higgs

model = LudwigModel(
    config='config_tabnet_transfer_santCustSat.yaml',
    logging_level=logging.INFO,
    backend="local",
)

higgs_df = higgs.load()
model.train(
    dataset=higgs_df,
    experiment_name='higgs_tabnet_transfer_santCustSat',
    model_name='higgs_tabnet_transfer_santCustSat',
    skip_save_processed_input=True
)
