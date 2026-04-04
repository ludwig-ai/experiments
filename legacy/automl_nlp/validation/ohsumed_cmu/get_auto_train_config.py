import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import ohsumed_cmu
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ohsumed_df = ohsumed_cmu.load(split=False)
ohsumed_cmu_df = get_repeatable_train_val_test_split(ohsumed_df, random_seed=42)

auto_config = create_auto_config(
    dataset=ohsumed_cmu_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
