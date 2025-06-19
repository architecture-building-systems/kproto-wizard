import sys
from pathlib import Path
import io
import zipfile
import pandas as pd
import streamlit as st
from datetime import datetime
from kprototypes_wizard.kproto_postprocessing import (
    generate_clustered_output,
    generate_cluster_overview
)

sys.path.append(str(Path(__file__).resolve().parents[2]))

def download_page():
    st.header("Download Clustered Dataset")

    if "pipeline_result" not in st.session_state or "selected_k" not in st.session_state:
        st.warning("Please run clustering and select a k value first.")
        return
    
        # Navigation + download
    back, _ = st.columns([1, 5])
    if back.button("← Back", use_container_width=True):
        st.session_state["current_page"] = "Review Results"
        st.rerun()

    selected_k = st.session_state["selected_k"]
    df_raw = st.session_state["df_raw"]
    pipeline_result = st.session_state["pipeline_result"]

    # Compute data
    df_clustered = generate_clustered_output(
        df_raw=df_raw,
        pipeline_result=pipeline_result,
        selected_k=selected_k
    )
    df_summary = generate_cluster_overview(df_clustered)

    # Explain contents
    with st.container(border=True):
        st.subheader("What's in the ZIP?")
        st.markdown("""
            :material/category: — Cluster overview (Summary table)
            
            :material/table_convert: — Clustered data (Assignments per input)
            
            :material/table: — Original Data (Unmodified upload)
                    """)
        # c1, c2, c3 = st.columns(3)
        # with c1: st.text("Cluster Overview\n(Summary table)")
        # with c2: st.text("Clustered Data\n(Assignments per input)")
        # with c3: st.text("Original Data\n(Unmodified upload)")


    # Choose filetype + download
    format_picker, download = st.columns([4, 2])
    with format_picker:
        format_choice = st.segmented_control(
        label="Download Format",
        options=["CSV", "XLSX"],
        key="download_format",
        label_visibility="collapsed",
        selection_mode="single",
        default="XLSX"
    )


    # Generate in-memory ZIP
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_choice == "CSV":
            zipf.writestr(f"cluster_overview_{timestamp}.csv", df_summary.to_csv(index=False))
            zipf.writestr(f"clustered_data_{timestamp}.csv", df_clustered.to_csv(index=False))
            zipf.writestr(f"original_data_{timestamp}.csv", df_raw.to_csv(index=False))

        elif format_choice == "XLSX":
            with io.BytesIO() as xlsx_buf:
                with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                    df_summary.to_excel(writer, sheet_name="Cluster Overview", index=False)
                    df_clustered.to_excel(writer, sheet_name="Clustered Data", index=False)
                    df_raw.to_excel(writer, sheet_name="Original Data", index=False)
                zipf.writestr(f"clustering_output_{timestamp}.xlsx", xlsx_buf.getvalue())

    # Finalize ZIP and offer download
    buffer.seek(0)
    with download:
        st.download_button(
            label="Download ZIP",
            data=buffer,
            file_name=f"clustering_results_{timestamp}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary",
            icon=":material/download:"
        )