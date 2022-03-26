import logging
import pprint

from load_util import load_bbcnews
from ludwig.automl import create_auto_config

bbcnews_df = load_bbcnews()

auto_config = create_auto_config(
    dataset=bbcnews_df,
    target='Category',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
