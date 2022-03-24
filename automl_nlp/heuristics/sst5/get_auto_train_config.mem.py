import logging
import pprint

from ludwig.datasets import sst5
from ludwig.automl import create_auto_config

sst5_df = sst5.load(split=False)

auto_config = create_auto_config(
    dataset=sst5_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
