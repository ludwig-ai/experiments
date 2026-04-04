import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import reuters_cmu
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

reuters_df = reuters_cmu.load(split=False)
reuters_cmu_df = get_repeatable_train_val_test_split(reuters_df, random_seed=42)

auto_config = create_auto_config(
    dataset=reuters_cmu_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
