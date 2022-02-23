import logging
import pprint

from ludwig.datasets import sst5
from ludwig.automl import auto_train

sst5_df = sst5.load(split=False)

auto_train_results = auto_train(
    dataset=sst5_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'text': {'word_sequence_length_limit': 44}}}
)

pprint.pprint(auto_train_results)
