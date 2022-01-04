import logging
import pprint

from ludwig.datasets import forest_cover
from ludwig.automl import create_auto_config

forest_cover_df = forest_cover.load()

auto_config = create_auto_config(
    dataset=forest_cover_df,
    target='Cover_Type',
    time_limit_s=7200,
    tune_for_memory=False,
    use_reference_config=True
)

pprint.pprint(auto_config)