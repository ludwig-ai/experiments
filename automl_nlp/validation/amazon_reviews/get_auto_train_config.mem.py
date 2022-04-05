import logging
import pprint

from load_util import load_amazon_reviews
from ludwig.automl import create_auto_config

amazon_reviews_df = load_amazon_reviews()

auto_config = create_auto_config(
    dataset=amazon_reviews_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
