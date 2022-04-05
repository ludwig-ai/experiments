import logging
import pprint

from ludwig.datasets import ohsumed
from ludwig.automl import create_auto_config

ohsumed_df = ohsumed.load(split=False)

auto_config = create_auto_config(
    dataset=ohsumed_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
