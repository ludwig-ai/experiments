import logging
import pprint

from load_util import load_amazon_review_polarity
from ludwig.automl import auto_train

amazon_review_polarity_df = load_amazon_review_polarity(include_title=True)

auto_train_results = auto_train(
    dataset=amazon_review_polarity_df,
    target='label',
    time_limit_s=18000,
    tune_for_memory=True,
)

pprint.pprint(auto_train_results)
