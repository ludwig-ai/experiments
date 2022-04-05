import logging
import pprint

from ludwig.datasets import imdb
from ludwig.automl import create_auto_config

imdb_df = imdb.load(split=False)

auto_config = create_auto_config(
    dataset=imdb_df,
    target='sentiment',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
