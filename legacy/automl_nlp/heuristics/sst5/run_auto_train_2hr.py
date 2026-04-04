import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import sst5
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

s_df = sst5.load(split=False)
sst5_df = get_repeatable_train_val_test_split(s_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=sst5_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
