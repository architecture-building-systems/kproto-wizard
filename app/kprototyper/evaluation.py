
import logging
logger = logging.getLogger(__name__)

import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score

def kprototypes_evaluate(df_encoded, categorical_indicies, k_range=(2,30), init='Huang', verbose=False, stop_on_failure=True):
    """
    Runs k-prototypes clustering over a range of k-values.
    
    Parameters:
        df_encoded (pd.Dataframe):      Fully encoded DataFrame (numerical+categorical).
        categorical_indicies (list):    Index positions of categorical columns in df_encoded.
        k_range (iterable):             Range of k-values to test (e.g., range(2,30)).
        init (str):                     Initialization method ('Huang' or 'Cao').
        verbose (bool):                 Whether to print log messages during iteration.
        stop_on_failure (bool):         Whether to stop evaluation on first ValueError.

        Returns:
            results (dict): {
                'costs': {k: cost},
                'silhouette_scores': {k: silhouette score},
                'assignments': {k: cluster labels (np.array)},
                'models': {k: trained KPrototypes model}
            }
    """

    # converts DataFrame to NumPy array
    x_in = df_encoded.to_numpy()

    costs = {}
    silhouette_scores = {}
    assignments = {}
    models = {}

    for k in k_range:
        try:
            model = KPrototypes(n_clusters=k, init=init, n_jobs=-2, random_state=0)
            clusters = model.fit_predict(x_in, categorical=categorical_indicies)

            costs[k] = model.cost_
            assignments[k] = clusters
            models[k] = model

            if k > 1 and isinstance(clusters, np.ndarray):
                silhouette_avg = silhouette_score(x_in, clusters)
                silhouette_scores[k] = silhouette_avg
            else:
                silhouette_scores[k] = np.nan

            if verbose:
                logger.info(f"[K={k}] Cost = {model.cost_:.2f}, Silhouette = {silhouette_scores[k]:.3f}")

        except ValueError as e:
            if stop_on_failure == True:
                logger.warning(f"[K={k}] Clustering Failed: {e}")
                # stop iterating if clustering cannot proceed
                break
            else:
                logger.warning(f"[K={k}] Clustering Failed: {e}")
                costs[k] = np.nan
                silhouette_scores[k] = np.nan

    return {
        'costs': costs,
        'silhouette_scores': silhouette_scores,
        'assignments': assignments,
        'models': models
    }