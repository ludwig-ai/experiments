import logging
import pprint

from load_util import load_walmart_recruiting
from ludwig.automl import auto_train

walmart_recruiting_df = load_walmart_recruiting()

auto_train_results = auto_train(
    dataset=walmart_recruiting_df,
    target='TripType',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
