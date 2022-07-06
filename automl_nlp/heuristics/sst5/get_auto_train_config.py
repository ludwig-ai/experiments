import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import sst5
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

s_df = sst5.load(split=False)
sst5_df = get_repeatable_train_val_test_split(s_df, 'label', random_seed=42)

auto_config = create_auto_config(
    dataset=sst5_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
