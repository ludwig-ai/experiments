import logging
import pprint

from load_util import load_goemotions
from ludwig.automl import auto_train

goemotions_df = load_goemotions()

auto_train_results = auto_train(
    dataset=goemotions_df,
    target='emotion_ids',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'emotion_ids', 'name': 'emotion_ids', 'type': 'set'}]}
)

pprint.pprint(auto_train_results)
