import logging
import pprint

from load_util import load_naval
from ludwig.automl import auto_train

naval_df = load_naval()

auto_train_results = auto_train(
    dataset=naval_df,
    target='gtcdsc',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
