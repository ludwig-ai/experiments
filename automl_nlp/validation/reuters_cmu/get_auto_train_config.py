import logging
import pprint

from ludwig.datasets import reuters
from ludwig.automl import create_auto_config

reuters_df = reuters.load(split=False)

auto_config = create_auto_config(
    dataset=reuters_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
