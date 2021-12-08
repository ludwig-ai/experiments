import logging

from ludwig.api import LudwigModel
from load_util import load_ames_housing

model = LudwigModel(
    config='config_tabnet_transfer_allstate.yaml',
    logging_level=logging.INFO,
    backend="local",
)

ames_housing_df = load_ames_housing()
model.train(
    dataset=ames_housing_df,
    experiment_name='ames_housing_tabnet_transfer_allstate',
    model_name='ames_housing_tabnet_transfer_allstate',
    skip_save_processed_input=True
)
