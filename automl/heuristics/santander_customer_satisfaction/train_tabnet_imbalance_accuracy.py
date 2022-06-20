import logging

from ludwig.api import LudwigModel
from load_util import load_santander_customer_satisfaction

model = LudwigModel(
    config='config_tabnet_imbalance_accuracy.yaml',
    logging_level=logging.INFO,
    backend="local",
)

santander_customer_satisfaction_df = load_santander_customer_satisfaction()
model.experiment(
    dataset=santander_customer_satisfaction_df,
    experiment_name='santander_customer_satisfaction_tabnet_imbalance_accuracy',
    model_name='santander_customer_satisfaction_tabnet_imbalance_accuracy',
    skip_save_processed_input=True
)
