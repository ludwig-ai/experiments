import logging
import pprint

from load_util import load_reuters_r8
from ludwig.automl import auto_train

reuters_r8_df = load_reuters_r8()

auto_train_results = auto_train(
    dataset=reuters_r8_df,
    target='intent',
    time_limit_s=3600,
    tune_for_memory=True,
    output_directory='s3://predibase-elotl/baseline/reuters_r8/',
)

pprint.pprint(auto_train_results)
