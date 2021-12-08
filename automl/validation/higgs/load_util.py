import pandas as pd

from ludwig.datasets import higgs

def load_higgs():
    higgs_df = higgs.load()
    if "split" in higgs_df.columns:
        train_df = higgs_df[higgs_df["split"] == 0]
        val_df = higgs_df[higgs_df["split"] == 1]
        test_df = higgs_df[higgs_df["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
            train_df = train_df.drop(val_df.index)

        val_df.split = 1
        higgs_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return higgs_df
