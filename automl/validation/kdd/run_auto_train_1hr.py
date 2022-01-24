import logging
import pprint

from load_util import load_kdd_appetency
from ludwig.automl import auto_train

kdd_appetency_df = load_kdd_appetency()

auto_train_results = auto_train(
    dataset=kdd_appetency_df,
    target='target',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
