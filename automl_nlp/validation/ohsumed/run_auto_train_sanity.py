import logging
import pprint

from load_util import load_ohsumed
from ludwig.automl import auto_train

ohsumed_df = load_ohsumed()

auto_train_results = auto_train(
    dataset=ohsumed_df,
    target='class',
    time_limit_s=360,
    tune_for_memory=False,
)

pprint.pprint(auto_train_results)
