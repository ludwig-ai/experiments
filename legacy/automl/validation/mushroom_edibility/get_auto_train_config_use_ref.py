import logging
import pprint

from ludwig.datasets import mushroom_edibility
from ludwig.automl import create_auto_config

mushroom_edibility_df = mushroom_edibility.load()

auto_config = create_auto_config(
    dataset=mushroom_edibility_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False,
    use_reference_config=True
)

pprint.pprint(auto_config)
