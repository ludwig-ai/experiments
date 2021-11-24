import logging
import pprint

from ludwig.datasets import santander_customer_transaction
from ludwig.automl import create_auto_config

santander_customer_transaction_df = santander_customer_transaction.load()

auto_config = create_auto_config(
    dataset=santander_customer_transaction_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
