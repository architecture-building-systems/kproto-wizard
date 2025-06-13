import pandas as pd

def generate_clustered_output(
    df_raw,
    pipeline_result,
    selected_k=None,
    output_path=None
):
    """
    Generates and optionally saves a clustered dataset with assigned cluster labels.

    Parameters:
        df_raw (pd.DataFrame): The original (unencoded) dataframe.
        pipeline_result (dict): Output from run_kprototypes_pipeline().
        selected_k (int or None): Cluster count to use (default = peak_k).
        output_path (str or None): If given, saves CSV to this path.

    Returns:
        pd.DataFrame: Cluster-labeled dataframe.
    """
    assignments_dict = pipeline_result["evaluation_results"]["assignments"]
    
    if selected_k is None:
        selected_k = pipeline_result["peak_k"]

    if selected_k not in assignments_dict or assignments_dict[selected_k] is None:
        raise ValueError(f"Cluster assignments for k={selected_k} are missing or invalid.")

    df_clustered = df_raw.copy()
    df_clustered[f"cluster_k{selected_k}"] = assignments_dict[selected_k]

    if output_path:
        df_clustered.to_csv(output_path, index=False)

    return df_clustered

def generate_cluster_overview(df_clustered):
    """
    Generates a summary table for each cluster, including cluster size,
    mean of numeric features, and mode of categorical features.

    Parameters:
        df_clustered (pd.DataFrame): The clustered dataset (must include a 'cluster_kX' column)

    Returns:
        pd.DataFrame: Overview of cluster characteristics sorted by size
    """
    # Identify cluster column
    cluster_col = [col for col in df_clustered.columns if col.startswith("cluster_k")]
    if not cluster_col:
        raise ValueError("No cluster label column found.")
    cluster_col = cluster_col[0]

    # Separate types
    non_cluster_cols = [col for col in df_clustered.columns if col not in [cluster_col, "name"]]
    num_cols = df_clustered[non_cluster_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df_clustered[non_cluster_cols].select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    cluster_summary = []

    for cluster_id, group in df_clustered.groupby(cluster_col):
        row = {"cluster": cluster_id, "count": len(group)}
        for col in num_cols:
            row[col] = round(group[col].mean(), 2)
        for col in cat_cols:
            row[col] = group[col].mode().iloc[0] if not group[col].mode().empty else None
        cluster_summary.append(row)

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.sort_values("count", ascending=False, inplace=True)
    summary_df.reset_index(drop=True, inplace=True)

    return summary_df
