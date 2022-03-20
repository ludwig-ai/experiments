import logging
import pprint

from ludwig.datasets import yelp_reviews
from ludwig.automl import create_auto_config

yelp_reviews_df = yelp_reviews.load(split=False)

auto_config = create_auto_config(
    dataset=yelp_reviews_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
