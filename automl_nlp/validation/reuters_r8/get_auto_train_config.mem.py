import logging
import pprint

from load_util import load_reuters_r8
from ludwig.automl import create_auto_config

reuters_r8_df = load_reuters_r8()

auto_config = create_auto_config(
    dataset=reuters_r8_df,
    target='intent',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
