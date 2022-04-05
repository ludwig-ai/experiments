import logging
import pprint

from ludwig.datasets import ohsumed_cmu
from ludwig.automl import create_auto_config

ohsumed_cmu_df = ohsumed_cmu.load(split=False)

auto_config = create_auto_config(
    dataset=ohsumed_cmu_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
