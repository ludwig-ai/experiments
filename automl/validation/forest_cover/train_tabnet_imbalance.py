import logging

from ludwig.api import LudwigModel
from ludwig.datasets import forest_cover

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

forest_cover_df = forest_cover.load(use_tabnet_split=True)
model.experiment(
    dataset=forest_cover_df,
    experiment_name='forest_cover_tabnet_imbalance',
    model_name='forest_cover_tabnet_imbalance',
    skip_save_processed_input=True
)
