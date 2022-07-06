import logging

from ludwig.api import LudwigModel
from ludwig.datasets import ames_housing
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_transfer_allstate.yaml',
    logging_level=logging.INFO,
    backend="local",
)

ames_df = ames_housing.load()
ames_housing_df = get_repeatable_train_val_test_split(ames_df, random_seed=42)
model.train(
    dataset=ames_housing_df,
    experiment_name='ames_housing_tabnet_transfer_allstate',
    model_name='ames_housing_tabnet_transfer_allstate',
    skip_save_processed_input=True
)
