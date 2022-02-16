import logging
import pprint

from ludwig.datasets import sst3
from ludwig.automl import auto_train

sst3_df = sst3.load(split=False)

auto_train_results = auto_train(
    dataset=sst3_df,
    target='label',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
