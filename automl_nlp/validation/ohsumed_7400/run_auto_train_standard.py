import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import ohsumed_7400
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ohsumed_df = ohsumed_7400.load(split=False)
ohsumed_df.drop("edge", axis=1, inplace=True)
ohsumed_7400_df = get_repeatable_train_val_test_split(ohsumed_df, 'intent', random_seed=42)

auto_train_results = auto_train(
    dataset=ohsumed_7400_df,
    target='intent',
    time_limit_s=3600,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
