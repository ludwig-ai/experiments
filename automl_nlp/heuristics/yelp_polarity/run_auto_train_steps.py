import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import yelp_review_polarity
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

yelp_df = yelp_review_polarity.load(split=False)
yelp_review_polarity_df = get_repeatable_train_val_test_split(yelp_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=yelp_review_polarity_df,
    target='label',
    time_limit_s=10800,
    tune_for_memory=True,
    user_config={'trainer': {'checkpoints_per_epoch': 2},
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
