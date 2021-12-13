import logging
import pprint

from load_util import load_higgs
from ludwig.automl import auto_train

higgs_df = load_higgs()

auto_train_results = auto_train(
    dataset=higgs_df,
    target='label',
    time_limit_s=14400,
    tune_for_memory=False,
    output_directory='s3://predibase-elotl/baseline/higgs/hours4/',
    user_config={'hyperopt': {'sampler': {'search_alg': {'type': 'random', 'max_concurrent': 3}}}}

)

pprint.pprint(auto_train_results)
