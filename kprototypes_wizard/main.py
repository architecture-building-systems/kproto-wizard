from kprototypes_wizard.kproto_preprocessing import prepare_user_csv
from kprototypes_wizard.kproto_run import run_kprototypes_pipeline
from kprototypes_wizard.kproto_postprocessing import generate_clustered_output   
from logging_config import configure_logging
import pandas as pd
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    configure_logging()

    # === 1. Load and analyze the CSV ===
    csv_path = "/Users/kirknewton/Documents/ETHZ/S4_S24/SemesterProject/_SP_Model/DB3_save/zh_train.csv"
    
    try:
        prep_result = prepare_user_csv(csv_path)
    except ValueError as e:
        logger.error(f"CSV preparation failed: {e}")
        exit(1)

    df = prep_result["dataframe"]
    inferred_num = prep_result["num_features"]
    inferred_cat = prep_result["cat_features"]

    logger.info(f"Inferred numerical features: {inferred_num}")
    logger.info(f"Inferred categorical features: {inferred_cat}")

    # === 2. Optionally override inferred values (replace manually here or via GUI input) ===
    num_features = inferred_num
    cat_features = inferred_cat

    # Example manual override (remove if you build a UI later)
    # num_features = ['b_era', 'b_com2_shp', 'b_dens2_far']
    # cat_features = ['u2_res', 'u2_nres', 'h1', 'w1']

    # === 3. Run the pipeline ===

    configure_logging()

    df = prep_result["dataframe"]

    result = run_kprototypes_pipeline(
        df_raw=df,
        num_features=['b_era', 'b_com2_shp', 'b_dens2_far'],
        cat_features=['u2_res', 'u2_nres', 'h1', 'w1'],
        k_range=range(2, 20),
        verbose=True
    )

    # === 4. Ask user for export cluster count (in GUI, this would be a dropdown or slider) ===
    selected_k = result["peak_k"]  # Or ask the user to override

    # === 5. Export the clustered dataframe ===
    clustered_df = generate_clustered_output(
        df_raw=df,
        pipeline_result=result,
        selected_k=selected_k,
        output_path=f"output_cluster_k{selected_k}.csv"
    )

    logger.info(f"Clustered dataset exported to output_cluster_k{selected_k}.csv")