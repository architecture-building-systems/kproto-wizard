import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from kprototypes_wizard.kproto_preprocessing import prepare_user_csv

import streamlit as st
import pandas as pd

def collect_current_assignments(df):
    return {
        col: st.session_state.get(f"assign_{col}", "off")
        for col in df.columns
    }

def save_column_settings(assignments):
    num = [k for k, v in assignments.items() if v == "numerical"]
    cat = [k for k, v in assignments.items() if v == "categorical"]
    st.session_state["num_features"] = num
    st.session_state["cat_features"] = cat
    st.session_state["col_assignments"] = assignments
    st.success(f"Saved: {len(num)} numerical and {len(cat)} categorical features.")

 
def preprocessing_page():

    df = st.session_state["df_raw"]

    st.title("Assign Column Types")
    st.caption("Review each column and assign it as categorical, numerical, or off.")

    if "df_raw" not in st.session_state:
        st.warning("Please upload a CSV file first.")
        return

    # Initialize column metadata only once
    if "prep_result" not in st.session_state:
        prep_result = prepare_user_csv(df)
        st.session_state["prep_result"] = prep_result

        # Initialize only if not yet present
        if "col_assignments" not in st.session_state:
            st.session_state["col_assignments"] = {
                col["name"]: col["inferred_type"] for col in prep_result["column_summary"]
            }

        if "column_info" not in st.session_state:
            st.session_state["column_info"] = {
                col["name"]: (col["dtype"], col["unique"]) for col in prep_result["column_summary"]
            }


    # Save button (top)
    if st.button("💾 Save Column Settings"):
        save_column_settings(collect_current_assignments(df))

    st.markdown("### Feature Assignment Table")

    # Simulated header row
    h1, h2, h3, h4 = st.columns([2, 1, 1, 5])
    h1.markdown("**Column**")
    h2.markdown("**Dtype**")
    h3.markdown("**Unique**")
    h4.markdown("**Assign Type**")

    # Main table rows
    for col_name in df.columns:
        dtype, n_unique = st.session_state["column_info"][col_name]
        default_type = st.session_state["col_assignments"].get(col_name, "off")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 5])
        with c1:
            st.markdown(f"**{col_name}**")
        with c2:
            st.markdown(f"`{dtype}`")
        with c3:
            st.markdown(f"`{n_unique}`")
        with c4:
            st.radio(
                label="",
                options=["categorical", "numerical", "off"],
                horizontal=True,
                key=f"assign_{col_name}",
                index=["categorical", "numerical", "off"].index(default_type)
            )

    st.divider()

    # Save button (bottom)
    if st.button("💾 Save Column Settings (Bottom)"):
        save_column_settings(collect_current_assignments(df))