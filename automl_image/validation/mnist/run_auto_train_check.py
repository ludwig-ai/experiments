import logging
import pprint

from load_util import load_mnist
from ludwig.automl import auto_train

mnist_df = load_mnist()

auto_train_results = auto_train(
    dataset=mnist_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
