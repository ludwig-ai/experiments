import logging
import pprint

from load_util import load_yelp_reviews
from ludwig.automl import auto_train

yelp_reviews_df = load_yelp_reviews()

auto_train_results = auto_train(
    dataset=yelp_reviews_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=True
)

pprint.pprint(auto_train_results)
