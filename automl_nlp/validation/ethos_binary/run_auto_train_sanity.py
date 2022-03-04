import logging
import pprint

from load_util import load_ethos_binary
from ludwig.automl import auto_train

ethos_binary_df = load_ethos_binary()

auto_train_results = auto_train(
    dataset=ethos_binary_df,
    target='isHate',
    time_limit_s=360,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
