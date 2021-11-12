import logging
import pprint

from ludwig.datasets import otto_group_product
from ludwig.automl import create_auto_config

otto_group_product_df = otto_group_product.load()

auto_config = create_auto_config(
    dataset=otto_group_product_df,
    target='target',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)
