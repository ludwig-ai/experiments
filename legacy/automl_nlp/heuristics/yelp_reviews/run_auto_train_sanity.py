import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import yelp_reviews
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

yelp_df = yelp_reviews.load(split=False)
yelp_reviews_df = get_repeatable_train_val_test_split(yelp_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=yelp_reviews_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
