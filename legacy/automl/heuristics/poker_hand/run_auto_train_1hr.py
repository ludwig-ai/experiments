import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import poker_hand
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

poker_df = poker_hand.load(split=False)
poker_hand_df = get_repeatable_train_val_test_split(poker_df, 'hand', random_seed=42)

auto_train_results = auto_train(
    dataset=poker_hand_df,
    target='hand',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
