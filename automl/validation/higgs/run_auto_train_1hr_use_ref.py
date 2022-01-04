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
    use_reference_config=True
)

pprint.pprint(auto_train_results)
