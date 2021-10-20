import logging

from ludwig.api import LudwigModel
from ludwig.datasets import synthetic_fraud

model = LudwigModel(
    config='config_tabnet_transfer_bnp.yaml',
    logging_level=logging.INFO,
    backend="local",
)

synthetic_fraud_df = synthetic_fraud.load()
model.train(
    dataset=synthetic_fraud_df,
    experiment_name='synthetic_fraud_tabnet_transfer_bnp',
    model_name='synthetic_fraud_tabnet_transfer_bnp',
    skip_save_processed_input=True
)
