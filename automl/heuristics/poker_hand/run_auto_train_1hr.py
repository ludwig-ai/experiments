import logging
import pprint

from load_util import load_poker_hand
from ludwig.automl import auto_train

poker_hand_df = load_poker_hand()

auto_train_results = auto_train(
    dataset=poker_hand_df,
    target='hand',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
