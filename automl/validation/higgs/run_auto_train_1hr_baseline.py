import logging
import pprint

from load_util import load_higgs
from ludwig.automl import auto_train

higgs_df = load_higgs()

auto_train_results = auto_train(
    dataset=higgs_df,
    target='label',
    time_limit_s=3600,
    tune_for_memory=False,
    output_directory='s3://predibase-elotl/baseline/higgs/hours1/',
)

pprint.pprint(auto_train_results)
