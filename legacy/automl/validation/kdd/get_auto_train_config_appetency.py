import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import kdd_appetency
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

kdd_df = kdd_appetency.load()
kdd_appetency_df = get_repeatable_train_val_test_split(kdd_df, random_seed=42)

auto_config = create_auto_config(
    dataset=kdd_appetency_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
