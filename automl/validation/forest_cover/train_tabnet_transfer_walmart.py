import logging

from ludwig.api import LudwigModel
from ludwig.datasets import forest_cover

model = LudwigModel(
    config='config_tabnet_transfer_walmart.yaml',
    logging_level=logging.INFO,
    backend="local",
)

forest_cover_df = forest_cover.load(use_tabnet_split=True)
model.train(
    dataset=forest_cover_df,
    experiment_name='forest_cover_tabnet_transfer_walmart',
    model_name='forest_cover_tabnet_transfer_walmart',
    skip_save_processed_input=True
)
