import pandas as pd

from ludwig.datasets import irony

def load_irony():
    irony_df = irony.load()

    if "split" not in irony_df.columns:
        irony_df["split"] = 0

    train_df = irony_df[irony_df["split"] == 0]
    val_df = irony_df[irony_df["split"] == 1]
    test_df = irony_df[irony_df["split"] == 2]

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

    irony_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return irony_df
