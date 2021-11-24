import logging
import pprint

from ludwig.datasets import mercedes_benz_greener
from ludwig.automl import create_auto_config

mercedes_benz_greener_df = mercedes_benz_greener.load()

auto_config = create_auto_config(
    dataset=mercedes_benz_greener_df,
    target='y',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
