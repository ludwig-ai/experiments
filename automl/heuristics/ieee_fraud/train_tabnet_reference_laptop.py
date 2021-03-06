import logging

from ludwig.api import LudwigModel
from ludwig.datasets import ieee_fraud

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

ieee_fraud_df = ieee_fraud.load()
model.train(
    dataset=ieee_fraud_df,
    experiment_name='ieee_fraud_tabnet_reference_laptop',
    model_name='ieee_fraud_tabnet_reference_laptop',
    skip_save_processed_input=True
)
