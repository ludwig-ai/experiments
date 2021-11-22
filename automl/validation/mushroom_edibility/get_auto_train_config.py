import logging
import pprint

from ludwig.datasets import mushroom_edibility
from ludwig.automl import create_auto_config

mushroom_edibility_df = mushroom_edibility.load()

auto_config = create_auto_config(
    dataset=mushroom_edibility_df,
    target='class',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)
