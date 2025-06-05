import logging
logger = logging.getLogger(__name__)

import pandas as pd

def get_categorical_indices(df, cat_features):
    """
    Given a dataframe and a list of categorical column names,
    returns their index positions in the dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame whose column order defines the indices.
        cat_features (list): List of categorical column names.

    Returns:
        List[int]: Indices of categorical columns.
    """
    return [df.columns.get_loc(col) for col in cat_features]