import logging
import pprint

from load_util import load_numerai28pt6
from ludwig.automl import auto_train

numerai28pt6_df = load_numerai28pt6()

auto_train_results = auto_train(
    dataset=numerai28pt6_df,
    target='target',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
