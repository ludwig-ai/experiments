import logging
import pprint

from load_util import load_sarcos
from ludwig.automl import auto_train

sarcos_df = load_sarcos()

auto_train_results = auto_train(
    dataset=sarcos_df,
    target='torque_1',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
