import logging
import pprint

from ludwig.datasets import ethos_binary
from ludwig.automl import create_auto_config

ethos_binary_df = ethos_binary.load()

auto_config = create_auto_config(
    dataset=ethos_binary_df,
    target='isHate',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
