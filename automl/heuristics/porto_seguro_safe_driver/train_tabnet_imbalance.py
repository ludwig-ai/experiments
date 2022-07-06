import logging

from ludwig.api import LudwigModel
from ludwig.datasets import porto_seguro_safe_driver
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

porto_df = porto_seguro_safe_driver.load()
porto_seguro_safe_driver_df = get_repeatable_train_val_test_split(porto_df, 'target', random_seed=42)
model.experiment(
    dataset=porto_seguro_safe_driver_df,
    experiment_name='porto_seguro_safe_driver_tabnet_imbalance',
    model_name='porto_seguro_safe_driver_tabnet_imbalance',
    skip_save_processed_input=True
)
