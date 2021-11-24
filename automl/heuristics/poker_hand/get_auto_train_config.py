import logging
import pprint

from ludwig.datasets import poker_hand
from ludwig.automl import create_auto_config

poker_hand_df, _, _ = poker_hand.load()

auto_config = create_auto_config(
    dataset=poker_hand_df,
    target='hand',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
