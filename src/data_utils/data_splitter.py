##### https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test

import pandas as pd
from sklearn.model_selection import train_test_split

def train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    """Create Train, Validation, Test Split

    Args:
        df_input (DataFrame)
        stratify_colname (str, optional): Defaults to 'y'.
        frac_train (float, optional): [Fraction of train split]. Defaults to 0.6.
        frac_val (float, optional): [Fraction of validation split. Defaults to 0.15.
        frac_test (float, optional): [Fraction of testing split]. Defaults to 0.25.
        random_state ([type], optional): Defaults to None.

    Returns:
        [DataFrame]: train df, validation df and test df
    """

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input
    y = df_input[[stratify_colname]]

    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)


    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test