import logging
import pprint

from ludwig.datasets import mushroom_edibility
from ludwig.automl import auto_train

mushroom_edibility_df = mushroom_edibility.load()

auto_train_results = auto_train(
    dataset=mushroom_edibility_df,
    target='class',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
