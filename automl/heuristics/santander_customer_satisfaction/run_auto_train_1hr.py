import logging
import pprint

from load_util import load_santander_customer_satisfaction
from ludwig.automl import auto_train

santander_customer_satisfaction_df = load_santander_customer_satisfaction()

auto_train_results = auto_train(
    dataset=santander_customer_satisfaction_df,
    target='TARGET',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
