import logging
import pprint

from load_util import load_amazon_review_polarity
from ludwig.automl import create_auto_config

amazon_review_polarity_df = load_amazon_review_polarity()

auto_config = create_auto_config(
    dataset=amazon_review_polarity_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
