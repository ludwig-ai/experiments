import logging

from ludwig.api import LudwigModel
from load_util import load_ieee_fraud

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

ieee_fraud_df = load_ieee_fraud()
model.experiment(
    dataset=ieee_fraud_df,
    experiment_name='ieee_fraud_tabnet_imbalance',
    model_name='ieee_fraud_tabnet_imbalance',
    skip_save_processed_input=True
)
