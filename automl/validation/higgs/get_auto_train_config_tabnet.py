import logging
import pprint

from ludwig.datasets import higgs
from ludwig.automl import create_auto_config

higgs_df = higgs.load()

auto_config = create_auto_config(
    dataset=higgs_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_specified_config={'combiner': {'type': 'tabnet'}}
)

pprint.pprint(auto_config)
