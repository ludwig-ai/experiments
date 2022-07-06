import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import bbcnews
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

b_df = bbcnews.load(split=False)
b_df.drop("ArticleId", axis=1, inplace=True)
bbcnews_df = get_repeatable_train_val_test_split(b_df, 'Category', random_seed=42)

auto_config = create_auto_config(
    dataset=bbcnews_df,
    target='Category',
    time_limit_s=7200,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
