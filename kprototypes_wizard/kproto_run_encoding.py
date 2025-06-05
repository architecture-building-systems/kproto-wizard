import logging
logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from kmodes.kmodes import encode_features


def encode_numerical(df, num_features, encoder=None):
    """ encodes numerical features using the provided encoder (default: StandardScaler) """
    if encoder is None:
        encoder = StandardScaler()

    # fail-safe: check for non-numeric data
    for col in num_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"[ERROR] Column '{col}' was passed as a numerical feature, but it contains non-numeric data. "
                f"Please check the column or reassign it to categorical features."
            )

    encoded = encoder.fit_transform(df[num_features])
    encoded_df = pd.DataFrame(encoded, columns=num_features, index=df.index)

    logger.info(f"Encoded {len(num_features)} numerical features.")
    return encoded_df, encoder

def encode_categorical(df, cat_features):
    """ encodes categorical features using kmodes integer encoder """

    # fail-safe: check for non-categorical data
    for col in cat_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"[ERROR] Column '{col}' was passed as a categorical feature, but it contains numeric data. "
                f"Please check whether it should be treated as a numerical feature."
            )

    cat_array = df[cat_features].values
    encoded_array, enc_map = encode_features(cat_array)

    # handle 1D case (single column)
    if encoded_array.ndim == 1:
        encoded_array = encoded_array.reshape(-1, 1)
    
    encoded_df = pd.DataFrame(encoded_array, columns=cat_features, index=df.index)

    logger.info(f"Encoded {len(cat_features)} categorical features.")
    return encoded_df, enc_map


def kprototypes_encode(df_raw, num_features, cat_features, num_encoder=None):
    '''
    Splits and encodes mixed-type dataframe for use with k-protoypes clustering, with type checks.

    Parameters:
        df_raw (pd.Dataframe):  Original mixed-type DataFrame
        num_features (list):    List of numerical feature column names
        cat_features (list):    List of categorical features column names
        num_encoder (sklearn-like): Encoder for numerical data (default: StandardScaler)

    Returns:
        df_encoded (pd.DataFrame):  Combined encoded DataFrame
        df_num_enc (pd.DataFrame):  Encoded numerical DataFrame
        df_cat_enc (pd.DataFrame):  Encoded categorical DataFrame
        cat_enc_map (dict):         Mapping from cagegorical labels to integers
    '''

    logger.info("Starting encoding of dataset...")

    df_num_enc, fitted_num_encoder = encode_numerical(df_raw, num_features, num_encoder)
    df_cat_enc, cat_enc_map = encode_categorical(df_raw, cat_features)

    df_encoded = pd.concat([df_num_enc, df_cat_enc], axis=1)
    
    logger.info("Encoding complete.")
    return df_encoded, df_num_enc, df_cat_enc, cat_enc_map