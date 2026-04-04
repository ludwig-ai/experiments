import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import ames_housing
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ames_df = ames_housing.load()
ames_housing_df = get_repeatable_train_val_test_split(ames_df, random_seed=42)

auto_config = create_auto_config(
    dataset=ames_housing_df,
    target='SalePrice',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
