import logging
import pprint

from load_util import load_amazon_review_polarity
from ludwig.automl import auto_train

amazon_review_polarity_df = load_amazon_review_polarity(include_title=True)

auto_train_results = auto_train(
    dataset=amazon_review_polarity_df,
    target='label',
    time_limit_s=18000,
    tune_for_memory=True,
    output_directory='s3://predibase-runs/nodeless/amazon_review_polarity/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}}},
)

pprint.pprint(auto_train_results)
