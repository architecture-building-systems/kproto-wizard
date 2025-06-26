import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from kprototypes_wizard.kproto_postprocessing import generate_clustered_output
from kprototypes_wizard.kproto_run_visualization import plot_kprototypes_results, detect_k_recommendations

def postprocessing_page():
    st.header("Review Results")

    result = st.session_state.get("pipeline_result", None)
    if result is None:
        st.warning("No clustering results found.")
        return

    # Navigation Pane
    back,action1,action2,action3,next = st.columns([1,1,2,1,1])
    if back.button("← Back", use_container_width=True):
        st.session_state["current_page"] = "Run Clustering"
        st.rerun()
    if next.button("Next →", 
                   type="primary", 
                   use_container_width=True, 
                   disabled=not st.session_state["clustering_run_success"]):
        st.session_state["current_page"] = "Download Output"
        st.rerun()

    peak_k, shoulder_k = detect_k_recommendations(result["evaluation_results"]["silhouette_scores"])
    k_range = list(result["evaluation_results"]["costs"].keys())
    slider = st.container(border=True)
    with slider:
        selected_k = st.slider("Select k for export", min_value=min(k_range), max_value=max(k_range), value=peak_k)
    st.session_state["selected_k"] = selected_k

    # Generate and display Plotly figure inline
    fig = plot_kprototypes_results(
        k_range=k_range,
        costs=result["evaluation_results"]["costs"],
        silhouettes=result["evaluation_results"]["silhouette_scores"],
        peak_k=peak_k,
        shoulder_k=shoulder_k,
        title="Silhouette & Cost Evaluation",
    )
    
    plot = st.container(border=True)
    plot.plotly_chart(fig, use_container_width=True)