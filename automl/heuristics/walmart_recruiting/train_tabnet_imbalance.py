import logging

from ludwig.api import LudwigModel
from ludwig.datasets import walmart_recruiting

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

walmart_recruiting_df = walmart_recruiting.load()
model.experiment(
    dataset=walmart_recruiting_df,
    experiment_name='walmart_recruiting_tabnet_imbalance',
    model_name='walmart_recruiting_tabnet_imbalance',
    skip_save_processed_input=True
)
