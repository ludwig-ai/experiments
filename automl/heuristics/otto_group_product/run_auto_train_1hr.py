import logging
import pprint

from load_util import load_otto_group_product
from ludwig.automl import auto_train

otto_group_product_df = load_otto_group_product()

auto_train_results = auto_train(
    dataset=otto_group_product_df,
    target='target',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
