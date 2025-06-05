import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from kprototypes_wizard.kproto_postprocessing import generate_clustered_output

def download_page():
    st.title("Download Clustered Dataset")

    if "pipeline_result" not in st.session_state or "selected_k" not in st.session_state:
        st.warning("Please run clustering and select a k value first.")
        return

    df_clustered = generate_clustered_output(
        df_raw=st.session_state["df_raw"],
        pipeline_result=st.session_state["pipeline_result"],
        selected_k=st.session_state["selected_k"]
    )

    csv = df_clustered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download Clustered Dataset (k={st.session_state['selected_k']})",
        data=csv,
        file_name=f"clustered_k{st.session_state['selected_k']}.csv",
        mime="text/csv"
    )
