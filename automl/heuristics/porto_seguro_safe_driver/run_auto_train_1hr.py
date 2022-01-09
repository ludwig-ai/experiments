import logging
import pprint

from load_util import load_porto_seguro_safe_driver
from ludwig.automl import auto_train

porto_seguro_safe_driver_df = load_porto_seguro_safe_driver()

auto_train_results = auto_train(
    dataset=porto_seguro_safe_driver_df,
    target='target',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
