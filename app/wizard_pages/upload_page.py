import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

def upload_page():
    st.title("Upload Your Dataset")
    st.markdown("""
    Upload a CSV or Excel file containing your building data. Requirements:
    - First column must be named **"name"** (case-sensitive)
    - No missing values
    - Only numeric and string-type columns
    - Reasonable row count (<= 5000 for MVP)
    """)

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format.")
                return

            # Check: First column must be named "name"
            if df.columns[0] != "name":
                st.error("The first column must be titled 'name' (case-sensitive). Please correct and re-upload.")
                return

            # Check for missing values
            if df.isnull().any().any():
                st.error("Your dataset contains missing values. Please clean it and re-upload.")
                return

            st.session_state["df_raw"] = df
            st.session_state["upload_verified"] = True
            st.success("File accepted! Continue to assign column types.")

        except Exception as e:
            st.error(f"Failed to process file: {e}")

    # Navigation Button (disabled unless upload successful)
    left, _, _, _, right = st.columns([1, 1, 2, 1, 1])
    if right.button(
        "Next →", type="primary", use_container_width=True,
        disabled=not st.session_state.get("upload_verified", False)):
        
        st.session_state["go_next"] = True
        if st.session_state.get("go_next"):
            st.session_state["current_page"] = "Assign Column Types"
            st.session_state["go_next"] = False  # reset
            st.rerun()