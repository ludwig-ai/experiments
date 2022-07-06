import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import naval
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

n_df = naval.load()
naval_df = get_repeatable_train_val_test_split(n_df, random_seed=42)

auto_train_results = auto_train(
    dataset=naval_df,
    target='gtcdsc',
    time_limit_s=14400,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
