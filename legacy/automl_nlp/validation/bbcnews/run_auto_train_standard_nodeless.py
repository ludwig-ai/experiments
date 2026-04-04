import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import bbcnews
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

b_df = bbcnews.load(split=False)
b_df.drop("ArticleId", axis=1, inplace=True)
bbcnews_df = get_repeatable_train_val_test_split(b_df, 'Category', random_seed=42)

auto_train_results = auto_train(
    dataset=bbcnews_df,
    target='Category',
    time_limit_s=3600,
    tune_for_memory=True,
    output_directory='s3://predibase-runs/nodeless/bbcnews/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}},
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
