import pandas as pd

def prepare_user_csv(df):
    """
    Loads dataframe, checks for NaNs, and infers column types.

    Parameters:
        df (str): Path to uploaded CSV file

    Returns:
        dict with keys:
            - dataframe: pd.DataFrame
            - num_features: list[str]
            - cat_features: list[str]
            - column_summary: list[dict] with keys: name, dtype, inferred_type
    Raises:
        ValueError: if any NaN/null values are present
    """

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values. Please clean or impute them before proceeding.")

    num_features = []
    cat_features = []
    column_summary = []

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()

        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 10:
            inferred_type = "numerical"
            num_features.append(col)
        else:
            inferred_type = "categorical"
            cat_features.append(col)

        column_summary.append({
            "name": col,
            "dtype": str(dtype),
            "unique": n_unique,
            "inferred_type": inferred_type
        })

    return {
        "dataframe": df,
        "num_features": num_features,
        "cat_features": cat_features,
        "column_summary": column_summary
    }