import logging

from ludwig.api import LudwigModel
from ludwig.datasets import forest_cover

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

forest_cover_df = forest_cover.load()
model.train(
    dataset=forest_cover_df,
    experiment_name='forest_cover_transf_sanity_laptop',
    model_name='forest_cover_transf_sanity_laptop',
    skip_save_processed_input=True
)
