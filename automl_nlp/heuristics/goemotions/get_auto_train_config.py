import logging
import pprint

from load_util import load_goemotions
from ludwig.automl import create_auto_config

goemotions_df = load_goemotions()

auto_config = create_auto_config(
    dataset=goemotions_df,
    target='emotion_ids',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'emotion_ids', 'name': 'emotion_ids', 'type': 'set'}]}
)

pprint.pprint(auto_config)
