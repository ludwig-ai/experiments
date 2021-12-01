import logging
import pprint

from ludwig.datasets import higgs
from ludwig.automl import auto_train

higgs_df = higgs.load()

auto_train_results = auto_train(
    dataset=higgs_df,
    target='label',
    time_limit_s=14400,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
