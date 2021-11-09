import logging

from ludwig.api import LudwigModel
from ludwig.datasets import sarcos
import pandas as pd

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

sarcos_df, _, _ = sarcos.load()
if "split" in sarcos_df.columns:
    train_df = sarcos_df[sarcos_df["split"] == 0]
    val_df = sarcos_df[sarcos_df["split"] == 1]
    test_df = sarcos_df[sarcos_df["split"] == 2]

    # no validation set provided, sample 10% of train set
    if len(val_df) == 0:
        val_df = train_df.sample(frac=0.1, replace=False)
        train_df = train_df.drop(val_df.index)

    val_df.split = 1
    sarcos_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

model.train(
    dataset=sarcos_df,
    experiment_name='sarcos_tabnet_reference_laptop',
    model_name='sarcos_tabnet_reference_laptop',
    skip_save_processed_input=True
)
