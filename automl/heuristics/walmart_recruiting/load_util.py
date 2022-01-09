import pandas as pd

from ludwig.datasets import walmart_recruiting

def load_walmart_recruiting():
    train_df = walmart_recruiting.load()

    # no validation set provided, sample 10% of train set
    val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
    train_df = train_df.drop(val_df.index)

    # no test set provided, sample 20% of train set
    test_df = train_df.sample(frac=0.2, replace=False, random_state=42)
    train_df = train_df.drop(test_df.index)

    train_df['split'] = 0
    val_df['split'] = 1
    test_df['split'] = 2
    walmart_recruiting_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return walmart_recruiting_df
