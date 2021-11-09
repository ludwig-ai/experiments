import logging

from ludwig.api import LudwigModel
from ludwig.datasets import walmart_recruiting

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

walmart_recruiting_df = walmart_recruiting.load()
model.train(
    dataset=walmart_recruiting_df,
    experiment_name='walmart_recruiting_tabnet_reference_laptop',
    model_name='walmart_recruiting_tabnet_reference_laptop',
    skip_save_processed_input=True
)
