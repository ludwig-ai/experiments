import logging
import pprint

from ludwig.datasets import otto_group_product
from ludwig.automl import auto_train

otto_group_product_df = otto_group_product.load()

auto_train_results = auto_train(
    dataset=otto_group_product_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
