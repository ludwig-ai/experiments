import logging
import numpy as np

from ludwig.api import LudwigModel
from ludwig.datasets import porto_seguro_safe_driver
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split
from eval_util import set_model_threshold, get_best_threshold


model = LudwigModel(
    config='config_tabnet_imbalance_ros.yaml',
    logging_level=logging.INFO,
    backend="local",
)

# Train model and evaluate on test split with default threshold
porto_df = porto_seguro_safe_driver.load()
porto_seguro_safe_driver_df = get_repeatable_train_val_test_split(porto_df, 'target', random_seed=42)
model.experiment(
    dataset=porto_seguro_safe_driver_df,
    experiment_name='porto_seguro_safe_driver_tabnet_imbalance_ros',
    model_name='porto_seguro_safe_driver_tabnet_imbalance_ros',
    skip_save_processed_input=True
)

# Get best threshold for model on validation split wrt specified metric
threshold_range = np.arange(0.0, 1.0, 0.05)
best_threshold = get_best_threshold(model, porto_seguro_safe_driver_df, 'target', 'avg_f1_score_macro', threshold_range)
print("Best threshold:", best_threshold)

# Using best threshold, evaluate model on test split
set_model_threshold(model, best_threshold)
model.evaluate(dataset=porto_seguro_safe_driver_df, split="test", collect_overall_stats=True)
