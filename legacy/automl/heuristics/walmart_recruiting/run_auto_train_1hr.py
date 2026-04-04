import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import walmart_recruiting
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

walmart_df = walmart_recruiting.load()
walmart_recruiting_df = get_repeatable_train_val_test_split(walmart_df, 'TripType', random_seed=42)

auto_train_results = auto_train(
    dataset=walmart_recruiting_df,
    target='TripType',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'TripType', 'name': 'TripType', 'type': 'category'}],
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
