import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import sst2
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

s_df = sst2.load(split=False)
sst2_df = get_repeatable_train_val_test_split(s_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=sst2_df,
    target='label',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
