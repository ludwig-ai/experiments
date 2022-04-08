import logging
import pprint

from load_util import load_ohsumed_7400
from ludwig.automl import auto_train

ohsumed_7400_df = load_ohsumed_7400()

auto_train_results = auto_train(
    dataset=ohsumed_7400_df,
    target='intent',
    time_limit_s=3600,
    tune_for_memory=True,
)

pprint.pprint(auto_train_results)
