import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import connect4
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

c_df = connect4.load()
connect4_df = get_repeatable_train_val_test_split(c_df, random_seed=42)

auto_config = create_auto_config(
    dataset=connect4_df,
    target='winner',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'winner', 'name': 'winner', 'type': 'category'}],
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
