import logging

from ludwig.api import LudwigModel
from ludwig.datasets import poker_hand
import pandas as pd

model = LudwigModel(
    config='config_tabnet_reference_auto.yaml',
    logging_level=logging.INFO,
    backend="local",
)

poker_hand_df, _, _ = poker_hand.load()
if "split" in poker_hand_df.columns:
    train_df = poker_hand_df[poker_hand_df["split"] == 0]
    val_df = poker_hand_df[poker_hand_df["split"] == 1]
    test_df = poker_hand_df[poker_hand_df["split"] == 2]

    # no validation set provided, sample 10% of train set
    if len(val_df) == 0:
        val_df = train_df.sample(frac=0.1, replace=False)
        train_df = train_df.drop(val_df.index)

    val_df.split = 1
    poker_hand_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

model.train(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_reference_auto',
    model_name='poker_hand_tabnet_reference_auto',
    skip_save_processed_input=True
)
