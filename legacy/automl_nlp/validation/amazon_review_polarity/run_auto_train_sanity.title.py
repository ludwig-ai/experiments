import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import amazon_review_polarity
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

amazon_df = amazon_review_polarity.load(split=False)
amazon_df["review_text"] = amazon_df["review_tile"] + " " + amazon_df["review_text"]
amazon_df.drop("review_tile", axis=1, inplace=True)
amazon_review_polarity_df = get_repeatable_train_val_test_split(amazon_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=amazon_review_polarity_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
