import pandas as pd

from ludwig.datasets import ohsumed_7400

def load_ohsumed_7400():
    ohsumed_7400_df = ohsumed_7400.load(split=False)

    ohsumed_7400_df.drop("edge", axis=1, inplace=True)

    if "split" in ohsumed_7400_df.columns:
        train_df = ohsumed_7400_df[ohsumed_7400_df["split"] == 0]
        val_df = ohsumed_7400_df[ohsumed_7400_df["split"] == 1]
        test_df = ohsumed_7400_df[ohsumed_7400_df["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
            train_df = train_df.drop(val_df.index)

        val_df.split = 1
        ohsumed_7400_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return ohsumed_7400_df
