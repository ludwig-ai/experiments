import logging
import pprint

from ludwig.datasets import naval
from ludwig.automl import create_auto_config

naval_df = naval.load()

auto_config = create_auto_config(
    dataset=naval_df,
    target='gtcdsc',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
