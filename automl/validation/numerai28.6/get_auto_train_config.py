import logging
import pprint

from ludwig.datasets import numerai28pt6
from ludwig.automl import create_auto_config

numerai28pt6_df = numerai28pt6.load()

auto_config = create_auto_config(
    dataset=numerai28pt6_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
