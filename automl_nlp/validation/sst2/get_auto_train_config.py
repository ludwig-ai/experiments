import logging
import pprint

from ludwig.datasets import sst2
from ludwig.automl import create_auto_config

sst2_df = sst2.load(split=False)

auto_config = create_auto_config(
    dataset=sst2_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
