import logging
import pprint

from load_util import load_reuters_cmu
from ludwig.automl import auto_train

reuters_cmu_df = load_reuters_cmu()

auto_train_results = auto_train(
    dataset=reuters_cmu_df,
    target='class',
    time_limit_s=360,
    tune_for_memory=False,
)

pprint.pprint(auto_train_results)
