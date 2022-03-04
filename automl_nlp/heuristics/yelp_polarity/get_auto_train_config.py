import logging
import pprint

from ludwig.datasets import yelp_review_polarity
from ludwig.automl import create_auto_config

yelp_polarity_df = yelp_review_polarity.load(split=False)

auto_config = create_auto_config(
    dataset=yelp_polarity_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
