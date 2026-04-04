import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import dbpedia
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

d_df = dbpedia.load(split=False)
d_df.drop("title", axis=1, inplace=True)
dbpedia_df = get_repeatable_train_val_test_split(d_df, 'label', random_seed=42)

auto_config = create_auto_config(
    dataset=dbpedia_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
