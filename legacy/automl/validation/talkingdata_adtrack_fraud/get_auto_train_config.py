import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import talkingdata_adtrack_fraud
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

talkingdata_df = talkingdata_adtrack_fraud.load()
talkingdata_adtrack_fraud_df = get_repeatable_train_val_test_split(talkingdata_df, 'is_attributed', random_seed=42)

auto_config = create_auto_config(
    dataset=talkingdata_adtrack_fraud_df,
    target='is_attributed',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
