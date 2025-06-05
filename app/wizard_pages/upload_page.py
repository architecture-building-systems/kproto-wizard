import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd

def upload_page():
    st.title("Upload Your Dataset")
    st.markdown("""
    Upload a CSV or Excel file containing your building data. Requirements:
    - No missing values
    - Only numeric and string-type columns
    - Reasonable row count (<= 5000 for MVP)
    """)

    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")

        if df.isnull().any().any():
            st.error("Your dataset contains missing values. Please clean it and re-upload.")
        else:
            st.session_state["df_raw"] = df
            st.success("File accepted! Continue to assign column types.")
