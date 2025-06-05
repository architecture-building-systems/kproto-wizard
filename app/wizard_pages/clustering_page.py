import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from kprototypes_wizard.kproto_run import run_kprototypes_pipeline

def clustering_page():
    st.title("Run K-Prototypes Clustering")

    if "df_raw" not in st.session_state or "num_features" not in st.session_state:
        st.warning("Please complete column assignment first.")
        return

    df = st.session_state["df_raw"]
    num_features = st.session_state["num_features"]
    cat_features = st.session_state["cat_features"]

    if st.button("Start Clustering"):
        st.success("Running... this may take a minute.")
        log_area = st.empty()
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            log_area.code("\n".join(log_lines), language="text")

        # Custom loop to manually call each K and log progress
        from kprototypes_wizard.kproto_run_evaluation import kprototypes_evaluate  # Or however you structure it
        from kprototypes_wizard.kproto_run_encoding import kprototypes_encode  # Assuming it's modular

        # Step 1: Encode
        df_enc, df_num_enc, df_cat_enc, cat_map = kprototypes_encode(
            df_raw=df,
            num_features=num_features,
            cat_features=cat_features
        )
        log("✅ Encoding complete.")

        # Step 2: Evaluate each k
        k_range = range(2, 20)
        results = {
            "costs": {},
            "silhouette_scores": {},
            "assignments": {},
            "models": {}
        }

        from kmodes.kprototypes import KPrototypes
        from sklearn.metrics import silhouette_score
        import numpy as np

        x_in = df_enc.to_numpy()
        cat_inds = list(range(len(cat_features)))

        for k in k_range:
            try:
                log(f"▶️  Trying k = {k}")
                model = KPrototypes(n_clusters=k, init="Huang", random_state=0, n_jobs=-1)
                labels = model.fit_predict(x_in, categorical=cat_inds)

                results["costs"][k] = model.cost_
                results["assignments"][k] = labels
                results["models"][k] = model

                if k > 1:
                    sil = silhouette_score(x_in, labels)
                    results["silhouette_scores"][k] = sil
                    log(f"   ✅ Success — Cost: {model.cost_:.2f}, Silhouette: {sil:.3f}")
                else:
                    results["silhouette_scores"][k] = np.nan
            except Exception as e:
                log(f"   ❌ Failed for k = {k}: {e}")
                break

        st.session_state["pipeline_result"] = {
            "evaluation_results": results
        }
        log("🎉 Clustering complete.")