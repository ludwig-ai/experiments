import logging
import pprint

from ludwig.datasets import connect4
from ludwig.automl import create_auto_config

connect4_df = connect4.load()

auto_config = create_auto_config(
    dataset=connect4_df,
    target='winner',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'winner', 'name': 'winner', 'type': 'category'}]}
)

pprint.pprint(auto_config)
