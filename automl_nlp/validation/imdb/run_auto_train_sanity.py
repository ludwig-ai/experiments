import logging
import pprint

from load_util import load_imdb
from ludwig.automl import auto_train

imdb_df = load_imdb()

auto_train_results = auto_train(
    dataset=imdb_df,
    target='sentiment',
    time_limit_s=360,
    tune_for_memory=False,
)

pprint.pprint(auto_train_results)
