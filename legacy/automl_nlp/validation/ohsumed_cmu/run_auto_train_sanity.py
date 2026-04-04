import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import ohsumed_cmu
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ohsumed_df = ohsumed_cmu.load(split=False)
ohsumed_cmu_df = get_repeatable_train_val_test_split(ohsumed_df, random_seed=42)

auto_train_results = auto_train(
    dataset=ohsumed_cmu_df,
    target='class',
    time_limit_s=360,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
