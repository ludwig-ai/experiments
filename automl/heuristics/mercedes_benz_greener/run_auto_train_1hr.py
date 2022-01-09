import logging
import pprint

from load_util import load_mercedes_benz_greener
from ludwig.automl import auto_train

mercedes_benz_greener_df = load_mercedes_benz_greener()

auto_train_results = auto_train(
    dataset=mercedes_benz_greener_df,
    target='y',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
