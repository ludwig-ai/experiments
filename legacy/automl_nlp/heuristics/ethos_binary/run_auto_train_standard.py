import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import ethos_binary
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ethos_df = ethos_binary.load()
ethos_binary_df = get_repeatable_train_val_test_split(ethos_df, 'isHate', random_seed=42)

auto_train_results = auto_train(
    dataset=ethos_binary_df,
    target='isHate',
    time_limit_s=3600,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
