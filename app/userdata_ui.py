import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

import pandas as pd
import streamlit as st
import io

from archetyper.session import ArchetyperSession
from kprototyper.session import KPrototyperSession
from shared.utils import export_database_to_zip

def show_user_data_page():
    st.markdown("## :material/folder: Your Data")

    # --- Refresh Button ---
    col1, col2 = st.columns([6, 1])
    col1.markdown("View your active and completed sessions below.")
    if col2.button("", key="refresh_user_data", help="Refresh", icon=":material/refresh:", use_container_width=True):
        st.rerun()

    # --- DatabaseMaker Sessions ---
    st.subheader(":material/database: DatabaseMaker Sessions")
    sessions = st.session_state.get("databasemaker__sessions", {})
    active_key = st.session_state.get("databasemaker__active_key", None)

    if not sessions:
        st.info("No DatabaseMaker sessions found.")
        return

    for key, session in sorted(sessions.items()):
        with st.container(border=True):

            header_container = st.container(border=False)
            info_container = st.container(border=False)
            
            header_container.markdown(f"###### {session.name}")
            
            if key == active_key:
                    header_container.success("Active Session")

            if session.download_ready and session.DB1 is not None:
                status_flag = ":green-badge[:material/check_box: Complete]"

                # --- Preview Final Construction Types ---
                with st.expander("Preview Construction Types", expanded=False):
                    if "construction_types" in session.DB1:
                        st.dataframe(session.DB1["construction_types"], use_container_width=True)
                    else:
                        st.warning("No construction_types table found.")

                # --- Downloads ---
                st.markdown("**Downloads:**")
                zip_buffer = export_database_to_zip(session.DB1)
                db_col, data_col = st.columns(2)

                db_col.download_button(
                    label="Download CEA Database (ZIP)",
                    icon=":material/database:",
                    data=zip_buffer.getvalue(),
                    file_name=f"{session.name}_cea_database.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary"
                )

                if session.clustered_df is not None and "cluster" in session.clustered_df.columns:
                    csv_buf = io.StringIO()
                    session.clustered_df.to_csv(csv_buf, index=False)
                    data_col.download_button(
                        label="Download Clustered Training Set (CSV)",
                        icon=":material/table:",
                        data=csv_buf.getvalue(),
                        file_name=f"{session.name}_clustered_training_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    data_col.info("Training data not available.")

            else:
                status_flag = ":orange-badge[:material/warning: Incomplete]"
                # Optional: Progress bar visualization from wizard (e.g. horizontal bar or step markers)
                step_list = ["Upload", "Assign Column Types", "Run Clustering", "Review Clustering", "Map Features", "Review Database", "Download"]
                current_step_index = step_list.index(session.step) if session.step in step_list else 0
                st.progress((current_step_index + 1) / len(step_list), text=f"Progress: Step {current_step_index + 1} of {len(step_list)}")


            info_container.markdown(f":gray-badge[Created: {session.created_at}] {status_flag}")

show_user_data_page()