import logging

from ludwig.api import LudwigModel
from ludwig.datasets import santander_customer_satisfaction
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_imbalance_accuracy.yaml',
    logging_level=logging.INFO,
    backend="local",
)

santander_df = santander_customer_satisfaction.load()
santander_customer_satisfaction_df = get_repeatable_train_val_test_split(santander_df, 'TARGET', random_seed=42)
model.experiment(
    dataset=santander_customer_satisfaction_df,
    experiment_name='santander_customer_satisfaction_tabnet_imbalance_accuracy',
    model_name='santander_customer_satisfaction_tabnet_imbalance_accuracy',
    skip_save_processed_input=True
)
