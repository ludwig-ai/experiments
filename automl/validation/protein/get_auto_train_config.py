import logging
import pprint

from ludwig.datasets import protein
from ludwig.automl import create_auto_config

protein_df = protein.load()

auto_config = create_auto_config(
    dataset=protein_df,
    target='RMSD',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
