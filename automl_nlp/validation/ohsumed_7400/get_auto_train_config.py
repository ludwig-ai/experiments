import logging
import pprint

from load_util import load_ohsumed_7400
from ludwig.automl import create_auto_config

ohsumed_7400_df = load_ohsumed_7400()

auto_config = create_auto_config(
    dataset=ohsumed_7400_df,
    target='intent',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
