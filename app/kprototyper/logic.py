import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import numpy as np

from kprototyper.session import KPrototyperSession

from kprototyper.encoding import kprototypes_encode
from kprototyper.evaluation import kprototypes_evaluate

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    if not uploaded_file.name.endswith((".csv", ".xls", ".xlsx")):
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")

    # Check: First column must be 'name'
    if df.columns[0] != "name":
        raise ValueError("The first column must be titled 'name' (case-sensitive).")

    # Check for missing values
    if df.isnull().any().any():
        raise ValueError("Your dataset contains missing values. Please clean it and re-upload.")

    return df


def preprocess_loaded_file(df):
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


def create_session_from_df(df: pd.DataFrame, session_id: str) -> KPrototyperSession:
    """
    Creates a new KPrototyperSession from a raw DataFrame.
    Automatically performs preprocessing to infer column types and metadata.
    """
    prep = preprocess_loaded_file(df)

    session = KPrototyperSession(name=session_id)
    session.input_data = prep["dataframe"]
    session.column_types = {
        col["name"]: col["inferred_type"] for col in prep["column_summary"]
    }
    if "name" in session.column_types:
        session.column_types["name"] = "off"

    session.column_metadata = {
        col["name"]: (col["dtype"], col["unique"]) for col in prep["column_summary"]
    }

    session.status = "ready for clustering"
    return session


def detect_k_recommendations(silhouette_scores: dict, threshold: float = 0.5):
    """
    Detects the peak and shoulder k-values from silhouette scores.
    
    Parameters:
        silhouette_scores (dict): Dictionary {k: silhouette_score}/
        threshold (float): Minimum silhouette score considered acceptable for 'shoulder' value

    Returns:
        tuple: (peak_k, shoulder_k)
    """

    # filter valid silhouette scores
    valid_scores = {k: score for k, score in silhouette_scores.items() if score is not None and not np.isnan(score)}

    if not valid_scores:
        return None, None
    
    # Peak: k with highest silhouette
    peak_k = max(valid_scores, key=valid_scores.get)

    # Shoulder: highest k with silhouette score above threshold 
    shoulder_candidates = [k for k, score in valid_scores.items() if score > threshold]
    shoulder_k = max(shoulder_candidates) if shoulder_candidates else None

    return peak_k, shoulder_k


def run_kprototypes_clustering(df_raw, column_types, k_range=(2, 30)):
    """
    Full pipeline for encoding, training, and evaluating k-prototypes clustering.
    Returns best model outputs and evaluation metrics.
    """
    # Step 1: Identify features by type
    selected_cols = [col for col, t in column_types.items() if t != "off"]
    num_features = [col for col in selected_cols if column_types[col] == "numerical"]
    cat_features = [col for col in selected_cols if column_types[col] == "categorical"]

    # Step 2: Encode data
    df_encoded, df_num, df_cat, cat_enc_map = kprototypes_encode(df_raw, num_features, cat_features)

    # Step 3: Get index positions of categorical columns in encoded dataframe
    cat_indices = list(range(len(num_features), len(num_features) + len(cat_features)))

    # Step 4: Run k-evaluation
    results = kprototypes_evaluate(df_encoded, categorical_indicies=cat_indices, k_range=range(*k_range), stop_on_failure=True)

    if not results["assignments"]:
        raise ValueError("Clustering failed for all values of k.")

    # Step 5: Select best k (lowest cost)
    best_k = min(results["costs"], key=lambda k: results["costs"][k])
    best_clusters = results["assignments"][best_k]

    # Step 6: Return clustered data
    clustered_df = df_raw.copy()
    clustered_df["cluster"] = best_clusters

    # Step 7: Modal values per cluster
    overview = clustered_df.groupby("cluster").agg(lambda x: x.mode().iloc[0] if not x.isnull().all() else None)
    overview.reset_index(inplace=True)

    # Step 8: Detect k reccomendations
    peak_k, shoulder_k = detect_k_recommendations(results["silhouette_scores"])

    return (
        clustered_df,
        overview,
        best_k,  # still based on cost
        results["costs"],
        results["silhouette_scores"],
        peak_k,
        shoulder_k
    )


def plot_kprototypes_results(
    k_range, costs, silhouettes, peak_k=None, shoulder_k=None,
    title='K-Prototypes Evaluation', threshold=0.5, width=1000, height=600
):
    x_vals = list(k_range)
    cost_vals = [costs.get(k, None) for k in x_vals]
    silhouette_vals = [silhouettes.get(k, None) for k in x_vals]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Cost curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=cost_vals, name='Cost', mode='lines+markers', line=dict(color='gray')
    ), secondary_y=False)

    # Silhouette curve
    fig.add_trace(go.Scatter(
        x=x_vals, y=silhouette_vals, name='Silhouette', mode='lines+markers', line=dict(color='#1471b0')
    ), secondary_y=True)

    # Horizontal threshold line
    fig.add_hline(y=threshold, line=dict(color='lightgray', dash='dot'), secondary_y=True)

    # Peak marker
    if peak_k:
        fig.add_trace(go.Scatter(
            x=[peak_k], y=[silhouettes[peak_k]], mode='markers+text',
            text=[f'Peak (k={peak_k})'], textposition='top center',
            marker=dict(size=10, color="#1471b0"), name='Peak'
        ), secondary_y=True)
        fig.add_vline(x=peak_k, line=dict(dash='dash', color='#1471b0'))


    # Shoulder marker
    if shoulder_k:
        fig.add_trace(go.Scatter(
            x=[shoulder_k], y=[silhouettes[shoulder_k]], mode='markers+text',
            text=[f'Shoulder (k={shoulder_k})'], textposition='bottom center',
            marker=dict(size=10, color="#1471b0"), name='Shoulder'
        ), secondary_y=True)
        fig.add_vline(x=shoulder_k, line=dict(dash='dash'))

    # Layout and formatting
    fig.update_layout(
        title=title,
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Cost (WSS)',
        yaxis2_title='Silhouette Score',
        template='plotly',  # <-- Better for dark mode
        width=width,
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(showgrid=False, secondary_y=False)
    fig.update_yaxes(showgrid=True, secondary_y=True)

    return fig
