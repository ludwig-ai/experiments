import logging
import pprint

from ludwig.datasets import higgs
from ludwig.automl import create_auto_config

higgs_df = higgs.load()

auto_config = create_auto_config(
    dataset=higgs_df,
    target='label',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)