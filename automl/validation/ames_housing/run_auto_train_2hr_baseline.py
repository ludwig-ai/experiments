import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import ames_housing
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ames_df = ames_housing.load()
ames_housing_df = get_repeatable_train_val_test_split(ames_df, random_seed=42)

auto_train_results = auto_train(
    dataset=ames_housing_df,
    target='SalePrice',
    time_limit_s=7200,
    tune_for_memory=False,
    output_directory='s3://predibase-elotl/baseline/ames_housing/hours2/',
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
