import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

def upload_page():
    st.header("Upload your building dataset")

    # Initialize persistent session flag
    if "upload_success" not in st.session_state:
        st.session_state["upload_success"] = False

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    uploder = st.container(border=True)
    with uploder:
        st.markdown("""
        Select a CSV or Excel file containing your building data for clustering. Requirements:
        - First column header is `id`
        - No `None` or `NaN` values
        - Only numeric and string-type columns
        """)



    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                st.session_state["upload_success"] = False
                st.error("Unsupported file format.")
                return

            if df.isnull().any().any():
                st.session_state["upload_success"] = False
                st.error("Your dataset contains missing values. Please clean it and re-upload.")
            else:
                st.session_state["df_raw"] = df
                st.session_state["upload_success"] = True
                st.success("File accepted! Continue to assign column types.")

        except Exception as e:
            st.session_state["upload_success"] = False
            st.error(f"Error reading file: {e}")

    # Navigation Pane
    back,action1,action2,action3,next = st.columns([1,1,2,1,1])
    if next.button("Next →", 
                   type="primary", 
                   use_container_width=True, 
                   disabled=not st.session_state["upload_success"]):
        st.session_state["current_page"] = "Assign Column Types"
        st.rerun()