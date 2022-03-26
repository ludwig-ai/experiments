import logging
import pprint

from load_util import load_bbcnews
from ludwig.automl import auto_train

bbcnews_df = load_bbcnews()

auto_train_results = auto_train(
    dataset=bbcnews_df,
    target='Category',
    time_limit_s=360,
    tune_for_memory=False,
)

pprint.pprint(auto_train_results)
