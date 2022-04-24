import logging
import pprint

from load_util import load_dbpedia
from ludwig.automl import auto_train

dbpedia_df = load_dbpedia()

auto_train_results = auto_train(
    dataset=dbpedia_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=True
)

pprint.pprint(auto_train_results)
