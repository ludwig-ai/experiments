import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import ethos_binary
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ethos_df = ethos_binary.load()
ethos_binary_df = get_repeatable_train_val_test_split(ethos_df, 'isHate', random_seed=42)

auto_config = create_auto_config(
    dataset=ethos_binary_df,
    target='isHate',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
