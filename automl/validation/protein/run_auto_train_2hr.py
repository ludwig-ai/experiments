import logging
import pprint

from load_util import load_protein
from ludwig.automl import auto_train

protein_df = load_protein()

auto_train_results = auto_train(
    dataset=protein_df,
    target='RMSD',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
