import logging
import pprint

from load_util import load_agnews
from ludwig.automl import auto_train

agnews_df = load_agnews()

auto_train_results = auto_train(
    dataset=agnews_df,
    target='class_index',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_train_results)
