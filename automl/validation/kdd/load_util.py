import pandas as pd

from ludwig.datasets import kdd_appetency

def load_kdd_appetency():
    kdd_appetency_df = kdd_appetency.load()
    if "split" in kdd_appetency_df.columns:
        train_df = kdd_appetency_df[kdd_appetency_df["split"] == 0]
        val_df = kdd_appetency_df[kdd_appetency_df["split"] == 1]
        test_df = kdd_appetency_df[kdd_appetency_df["split"] == 2]

        # no test set provided, sample 20% of train set
        if len(test_df) == 0:
            test_df = train_df.sample(frac=0.2, replace=False, random_state=42)
            train_df = train_df.drop(test_df.index)
        test_df.split = 2

        kdd_appetency_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return kdd_appetency_df
