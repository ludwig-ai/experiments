import logging
import pprint

from ludwig.datasets import sst2
from ludwig.automl import auto_train

sst2_df = sst2.load(split=False)

auto_train_results = auto_train(
    dataset=sst2_df,
    target='label',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
