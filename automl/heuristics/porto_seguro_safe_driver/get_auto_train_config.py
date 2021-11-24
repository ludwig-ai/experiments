import logging
import pprint

from ludwig.datasets import porto_seguro_safe_driver
from ludwig.automl import create_auto_config

porto_seguro_safe_driver_df = porto_seguro_safe_driver.load()

auto_config = create_auto_config(
    dataset=porto_seguro_safe_driver_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
