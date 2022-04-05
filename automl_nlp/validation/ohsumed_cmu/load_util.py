import pandas as pd

from ludwig.datasets import ohsumed_cmu

def load_ohsumed_cmu():
    ohsumed_cmu_df = ohsumed_cmu.load()

    if "split" not in ohsumed_cmu_df.columns:
        ohsumed_cmu_df["split"] = 0

    train_df = ohsumed_cmu_df[ohsumed_cmu_df["split"] == 0]
    val_df = ohsumed_cmu_df[ohsumed_cmu_df["split"] == 1]
    test_df = ohsumed_cmu_df[ohsumed_cmu_df["split"] == 2]

    # no validation set provided, sample 10% of train set
    if len(val_df) == 0:
        val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
        train_df = train_df.drop(val_df.index)
    val_df.split = 1

    # no test set provided, sample 20% of train set
    if len(test_df) == 0:
        test_df = train_df.sample(frac=0.2, replace=False, random_state=42)
        train_df = train_df.drop(test_df.index)
    test_df.split = 2

    ohsumed_cmu_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return ohsumed_cmu_df