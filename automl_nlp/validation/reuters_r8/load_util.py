import pandas as pd

from ludwig.datasets import reuters_r8

def load_reuters_r8():
    reuters_r8_df = reuters_r8.load(split=False)

    reuters_r8_df.drop("edge", axis=1, inplace=True)

    if "split" in reuters_r8_df.columns:
        train_df = reuters_r8_df[reuters_r8_df["split"] == 0]
        val_df = reuters_r8_df[reuters_r8_df["split"] == 1]
        test_df = reuters_r8_df[reuters_r8_df["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
            train_df = train_df.drop(val_df.index)

        val_df.split = 1
        reuters_r8_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return reuters_r8_df
