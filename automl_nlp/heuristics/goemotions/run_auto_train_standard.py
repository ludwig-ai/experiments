import logging
import pprint

from load_util import load_goemotions
from ludwig.automl import auto_train

goemotions_df = load_goemotions()

auto_train_results = auto_train(
    dataset=goemotions_df,
    target='emotion_ids',
    time_limit_s=10800,
    tune_for_memory=True
)

pprint.pprint(auto_train_results)
