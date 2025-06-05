import logging
logger = logging.getLogger(__name__)

from kprototypes_wizard.kproto_run_encoding import kprototypes_encode
from kprototypes_wizard.kproto_run_evaluation import kprototypes_evaluate
from kprototypes_wizard.kproto_run_visualization import plot_kprototypes_results, detect_k_recommendations
from kprototypes_wizard.kproto_run_utils import get_categorical_indices

def run_kprototypes_pipeline(
    df_raw,
    num_features,
    cat_features,
    k_range=range(2, 30),
    num_encoder=None,
    init='Huang',
    silhouette_threshold=0.5,
    verbose=False,
    stop_on_failure=True,
    plot_title='K-Prototypes Evaluation'
):
    """
    Full pipeline for encoding, clustering, evaluation, and visualization.
    """

    logger.info("Starting k-prototypes pipeline...")

    # Step 1: Encode
    logger.info("Step 1: Encoding input dataframe...")
    
    df_encoded, df_num_enc, df_cat_enc, cat_map = kprototypes_encode(
        df_raw, num_features, cat_features, num_encoder=num_encoder
    )

    # Step 2: Get categorical indices
    logger.info("Step 2: Extracting categorical indicies...")  

    categorical_indices = get_categorical_indices(df_encoded, cat_features)

    # Step 3: Evaluate
    logger.info("Step 3: Running k-prototypes evaluation...")  

    results = kprototypes_evaluate(
        df_encoded,
        categorical_indices,
        k_range=k_range,
        init=init,
        verbose=verbose,
        stop_on_failure=stop_on_failure
    )

    # Step 4: Suggest k values
    logger.info("Step 4: Suggesting k-values...") 
    peak_k, shoulder_k = detect_k_recommendations(results['silhouette_scores'], threshold=silhouette_threshold)

    # Step 5: Plot
    logger.info("Step 5: Plotting results...") 
    plot_kprototypes_results(
        k_range=k_range,
        costs=results['costs'],
        silhouettes=results['silhouette_scores'],
        peak_k=peak_k,
        shoulder_k=shoulder_k,
        title=plot_title,
        threshold=silhouette_threshold
    )

    return {
        'encoded_data': df_encoded,
        'evaluation_results': results,
        'peak_k': peak_k,
        'shoulder_k': shoulder_k
    }
