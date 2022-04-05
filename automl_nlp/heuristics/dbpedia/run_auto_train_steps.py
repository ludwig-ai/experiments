import logging
import pprint

from load_util import load_dbpedia
from ludwig.automl import auto_train

dbpedia_df = load_dbpedia()

auto_train_results = auto_train(
    dataset=dbpedia_df,
    target='label',
    time_limit_s=10800,
    tune_for_memory=True,
    user_config={'trainer': {'checkpoints_per_epoch': 2}}
)

pprint.pprint(auto_train_results)