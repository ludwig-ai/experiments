import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import walmart_recruiting
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

walmart_df = walmart_recruiting.load()
walmart_recruiting_df = get_repeatable_train_val_test_split(walmart_df, 'TripType', random_seed=42)

auto_config = create_auto_config(
    dataset=walmart_recruiting_df,
    target='TripType',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'TripType', 'name': 'TripType', 'type': 'category'}],
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
