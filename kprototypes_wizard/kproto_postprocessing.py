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