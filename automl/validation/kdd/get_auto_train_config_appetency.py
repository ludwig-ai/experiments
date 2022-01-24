import logging
import pprint

from ludwig.datasets import kdd_appetency
from ludwig.automl import create_auto_config

kdd_appetency_df = kdd_appetency.load()

auto_config = create_auto_config(
    dataset=kdd_appetency_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
