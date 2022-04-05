import logging
import pprint

from load_util import load_ohsumed_cmu
from ludwig.automl import auto_train

ohsumed_cmu_df = load_ohsumed_cmu()

auto_train_results = auto_train(
    dataset=ohsumed_cmu_df,
    target='class',
    time_limit_s=360,
    tune_for_memory=False,
)

pprint.pprint(auto_train_results)
