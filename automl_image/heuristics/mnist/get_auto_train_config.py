import logging
import pprint

from load_util import load_mnist
from ludwig.automl import create_auto_config

mnist_df = load_mnist()

auto_config = create_auto_config(
    dataset=mnist_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
