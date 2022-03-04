import logging
import pprint

from load_util import load_yelp_review_polarity
from ludwig.automl import auto_train

yelp_review_polarity_df = load_yelp_review_polarity()

auto_train_results = auto_train(
    dataset=yelp_review_polarity_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
