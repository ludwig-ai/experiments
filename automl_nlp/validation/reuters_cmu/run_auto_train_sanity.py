import logging
import pprint

from load_util import load_reuters
from ludwig.automl import auto_train

reuters_df = load_reuters()

auto_train_results = auto_train(
    dataset=reuters_df,
    target='class',
    time_limit_s=360,
    tune_for_memory=False,
)

pprint.pprint(auto_train_results)
