import logging
import pprint

from ludwig.datasets import sst3
from ludwig.automl import create_auto_config

sst3_df = sst3.load(split=False)

auto_config = create_auto_config(
    dataset=sst3_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
