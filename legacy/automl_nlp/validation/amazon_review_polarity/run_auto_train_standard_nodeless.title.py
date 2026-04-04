import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import amazon_review_polarity
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

amazon_df = amazon_review_polarity.load(split=False)
amazon_df["review_text"] = amazon_df["review_tile"] + " " + amazon_df["review_text"]
amazon_df.drop("review_tile", axis=1, inplace=True)
amazon_review_polarity_df = get_repeatable_train_val_test_split(amazon_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=amazon_review_polarity_df,
    target='label',
    time_limit_s=18000,
    tune_for_memory=True,
    output_directory='s3://predibase-runs/nodeless/amazon_review_polarity/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}},
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
