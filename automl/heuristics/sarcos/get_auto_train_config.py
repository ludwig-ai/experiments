import logging
import pprint

from ludwig.datasets import sarcos
from ludwig.automl import create_auto_config

sarcos_df, _, _ = sarcos.load()

auto_config = create_auto_config(
    dataset=sarcos_df,
    target='torque_1',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
