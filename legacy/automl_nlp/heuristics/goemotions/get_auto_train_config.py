import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import goemotions
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

go_df = goemotions.load(split=False)
go_df.drop("comment_id", axis=1, inplace=True)
goemotions_df = get_repeatable_train_val_test_split(go_df, 'emotion_ids', random_seed=42)

auto_config = create_auto_config(
    dataset=goemotions_df,
    target='emotion_ids',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'emotion_ids', 'name': 'emotion_ids', 'type': 'set'}],
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
