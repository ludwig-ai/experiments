import logging
import pprint

from load_util import load_agnews
from ludwig.automl import create_auto_config

agnews_df = load_agnews()

auto_config = create_auto_config(
    dataset=agnews_df,
    target='class_index',
    time_limit_s=7200,
    tune_for_memory=True
)

pprint.pprint(auto_config)
