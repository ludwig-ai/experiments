import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import amazon_reviews
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

amazon_df = amazon_reviews.load(split=False)
amazon_df.drop("review_tile", axis=1, inplace=True)
amazon_reviews_df = get_repeatable_train_val_test_split(amazon_df, 'label', random_seed=42)

auto_config = create_auto_config(
    dataset=amazon_reviews_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
