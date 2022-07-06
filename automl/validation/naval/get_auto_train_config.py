import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import naval
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

n_df = naval.load()
naval_df = get_repeatable_train_val_test_split(n_df, random_seed=42)

auto_config = create_auto_config(
    dataset=naval_df,
    target='gtcdsc',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
