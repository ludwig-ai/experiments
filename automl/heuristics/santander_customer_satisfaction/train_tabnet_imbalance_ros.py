import logging

from ludwig.api import LudwigModel
from ludwig.datasets import santander_customer_satisfaction

model = LudwigModel(
    config='config_tabnet_imbalance_ros.yaml',
    logging_level=logging.INFO,
    backend="local",
)

santander_customer_satisfaction_df = santander_customer_satisfaction.load()
model.experiment(
    dataset=santander_customer_satisfaction_df,
    experiment_name='santander_customer_satisfaction_tabnet_imbalance_ros',
    model_name='santander_customer_satisfaction_tabnet_imbalance_ros',
    skip_save_processed_input=True
)
