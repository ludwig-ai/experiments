import logging
import pprint

from load_util import load_mushroom_edibility
from ludwig.automl import auto_train

mushroom_edibility_df = load_mushroom_edibility()

auto_train_results = auto_train(
    dataset=mushroom_edibility_df,
    target='class',
    time_limit_s=3600,
    tune_for_memory=False,
    output_directory='s3://predibase-elotl/nodeless/mushroom_edibility/hours1/',
)

pprint.pprint(auto_train_results)
