import logging
import pprint

from load_util import load_dbpedia
from ludwig.automl import create_auto_config

dbpedia_df = load_dbpedia()

auto_config = create_auto_config(
    dataset=dbpedia_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
