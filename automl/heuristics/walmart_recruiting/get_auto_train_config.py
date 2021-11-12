import logging
import pprint

from ludwig.datasets import walmart_recruiting
from ludwig.automl import create_auto_config

walmart_recruiting_df = walmart_recruiting.load()

auto_config = create_auto_config(
    dataset=walmart_recruiting_df,
    target='TripType',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)
