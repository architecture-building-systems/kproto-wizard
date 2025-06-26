import streamlit as st
from kprototyper.session import KPrototyperSession
from kprototyper.logic import load_uploaded_file, preprocess_loaded_file, run_kprototypes_clustering, plot_kprototypes_results, create_session_from_df 
import uuid
import pandas as pd
from plotly.subplots import make_subplots
import io


# Define wizard steps
KPROTO_STEPS = ["Upload", "Assign Column Types", "Run Clustering", "Review & Download"]

# Initialize navigation state
if "kproto_step" not in st.session_state:
    st.session_state.kproto_step = "Upload"
if "kproto_sessions" not in st.session_state:
    st.session_state.kproto_sessions = []

if "kproto_step_success" not in st.session_state:
    st.session_state.kproto_step_success = {
        step: False for step in KPROTO_STEPS
    }


# -------------------------------
# Step UI Functions
# -------------------------------

def navigation_bar(step_index):
    step = KPROTO_STEPS[step_index]
    progress_state = st.session_state.kproto_step_success.get(step, False)

    nav_container = st.container(border=False)
    with nav_container:
        cols = st.columns([4, 1, 1])
        with cols[0]:
            st.markdown(f"#### Step {step_index + 1}: {step}")
        if step_index > 0 and cols[1].button("← Back", use_container_width=True):
            st.session_state.kproto_step = KPROTO_STEPS[step_index - 1]
            st.rerun()
        if step_index < len(KPROTO_STEPS) - 1 and cols[2].button("Next →", use_container_width=True, disabled=not progress_state, type="primary"):
            st.session_state.kproto_step = KPROTO_STEPS[step_index + 1]
            st.rerun()


def show_upload_ui(step_index):
    navigation_bar(step_index)

    # Initialize the control flag (only once)
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # Do nothing if already successful
    if uploaded_file and not st.session_state.upload_success:
        try:
            df = load_uploaded_file(uploaded_file)
            st.success("File validated successfully.")
            st.dataframe(df.head())

            session_id = f"run_{uuid.uuid4().hex[:6]}"
            session = create_session_from_df(df, session_id=session_id)
            st.session_state.kproto_sessions.append(session)

            # Set flags
            st.session_state.upload_success = True
            st.session_state.kproto_step_success["Upload"] = True

            st.success(f"Session '{session_id}' created with inferred column types.")
            st.rerun()

        except ValueError as e:
            st.error(str(e))


def show_column_typing_ui(step_index):
    navigation_bar(step_index)

    if not st.session_state.kproto_sessions:
        st.warning("No active session. Please upload a dataset first.")
        return

    session = st.session_state.kproto_sessions[-1]
    df = session.input_data

    if df is None:
        st.warning("No input data found in session.")
        return

    # Infer column types if not yet assigned
    if session.column_types is None:
        metadata = preprocess_loaded_file(df)
        inferred = {col["name"]: col["inferred_type"] for col in metadata["column_summary"]}
        if "name" in inferred:
            inferred["name"] = "off"
        session.column_types = inferred

        # Save dtype/unique info for display
        session.column_metadata = {
            col["name"]: (col["dtype"], col["unique"]) for col in metadata["column_summary"]
        }

    # Save button and next-step handling
    table = st.container(border=True)
    with table:
        table_title, table_save = st.columns([4,2])
        with table_title:
            st.markdown("#### Assign a type to each column")
        with table_save:
            if st.button("Save Assignments", icon=":material/save:", use_container_width=True):
                st.success(f"Saved: {sum(v == 'numerical' for v in session.column_types.values())} numerical and {sum(v == 'categorical' for v in session.column_types.values())} categorical columns.")
                st.session_state.kproto_step_success["Assign Column Types"] = True
                st.rerun()

        # Simulated header row
        h1, h2, h3, h4 = st.columns([2, 1, 1, 3])
        h1.markdown("**Column**")
        h2.markdown("**Dtype**")
        h3.markdown("**Unique**")
        h4.markdown("**Assign Type**")

        for col in df.columns:
            dtype, n_unique = session.column_metadata.get(col, ("unknown", "?"))
            default_type = session.column_types.get(col, "off")

            c1, c2, c3, c4 = st.columns([2, 1, 1, 3])
            with c1:
                st.markdown(f"**{col}**")
            with c2:
                st.markdown(f"`{dtype}`")
            with c3:
                st.markdown(f"`{n_unique}`")
            with c4:
                session.column_types[col] = st.segmented_control(
                    label="Type",
                    label_visibility="collapsed",
                    options=["categorical", "numerical", "off"],
                    format_func=lambda x: x,
                    selection_mode="single",
                    key=f"type_{col}",
                    default=default_type,
                    disabled=(col == "name")
                    # on_change=st.session_state.kproto_step_success(step, False)
                )





def show_clustering_ui(step_index):
    navigation_bar(step_index)

    if not st.session_state.kproto_sessions:
        st.warning("No session found. Please upload and configure data.")
        return

    session = st.session_state.kproto_sessions[-1]

    if session.input_data is None or session.column_types is None:
        st.warning("Missing input data or column types.")
        return

    if st.button("Run Clustering"):
        try:
            clustered, overview, best_k, cost_dict, sil_dict, peak_k, shoulder_k = run_kprototypes_clustering(
                session.input_data,
                session.column_types,
                k_range=(2, 30)
            )

            session.set_output(
                clustered_df=clustered,
                overview_df=overview,
                k=best_k,
                cost_dict=cost_dict,
                sil_dict=sil_dict
            )

            session.peak_k = peak_k
            session.shoulder_k = shoulder_k



            st.success(f"Clustering complete. Best k = {best_k}")
            st.dataframe(overview)

        except Exception as e:
            st.error(f"Clustering failed: {e}")

    elif session.is_complete():
        st.success(f"Clustering already completed for k = {session.selected_k}")
        st.dataframe(session.cluster_overview)
        st.session_state.kproto_step_success["Run Clustering"] = True
        st.rerun()


def show_postprocessing_ui(step_index):
    navigation_bar(step_index)

    if not st.session_state.kproto_sessions:
        st.warning("No session found.")
        return

    session = st.session_state.kproto_sessions[-1]

    if not session.is_complete():
        st.warning("Clustering not yet completed.")
        return

    st.markdown(f"#### Best k = {session.selected_k}")

    # --- Visualization: Cost & Silhouette ---
    st.markdown("#### Clustering Evaluation")

    if session.cost_per_k:
        st.markdown("#### Evaluation Chart")
        fig = plot_kprototypes_results(
            k_range=session.cost_per_k.keys(),
            costs=session.cost_per_k,
            silhouettes=session.silhouette_per_k,
            peak_k=session.peak_k,
            shoulder_k=session.shoulder_k
        )
    st.plotly_chart(fig, use_container_width=True)

    # --- Select alternative k (optional re-run) ---
    st.markdown("#### Optionally select a different k")
    available_ks = sorted(session.cost_per_k.keys())

    suggested_k = session.peak_k # or session.shoulder_k or session.selected_k

    selected_k = st.slider(
        label="Select k value", min_value=min(available_ks), max_value=max(available_ks), value=suggested_k)

    # --- Download options ---
    st.markdown("#### Download Cluster Results")

    download_format = st.radio("Download format", ["CSV", "Excel"], horizontal=True)

    def convert_to_excel(df_dict):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            for name, df in df_dict.items():
                df.to_excel(writer, sheet_name=name, index=False)
        buffer.seek(0)
        return buffer

    df_dict = {
        "clustered_data": session.clustered_data,
        "cluster_overview": session.cluster_overview
    }

    if download_format == "CSV":
        st.download_button("Download Clustered Data (CSV)",
                           data=session.clustered_data.to_csv(index=False),
                           file_name=f"{session.name}_clustered.csv",
                           mime="text/csv")

        st.download_button("Download Cluster Overview (CSV)",
                           data=session.cluster_overview.to_csv(index=False),
                           file_name=f"{session.name}_overview.csv",
                           mime="text/csv")

    else:  # Excel
        excel_file = convert_to_excel(df_dict)
        st.download_button("Download All Results (XLSX)",
                           data=excel_file,
                           file_name=f"{session.name}_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Finalize session ---
    st.markdown("#### Finalize and Save")

    if st.button("✅ Finalize Session"):
        session.status = "finalized"
        st.success("Session marked as finalized and ready for archetyping.")


# -------------------------------
# Layout and Navigation
# -------------------------------

st.markdown(f"## K-Prototyper")
step_index = KPROTO_STEPS.index(st.session_state.kproto_step)

# Navigation Pane
# cols = st.columns([4,2,2])
# st.markdown(f"#### {st.session_state.kproto_step}")
# if step_index > 0 and cols[0].button("← Back"):
#     st.session_state.kproto_step = KPROTO_STEPS[step_index - 1]
# if step_index < len(KPROTO_STEPS) - 1 and cols[1].button("Next →"):
#     st.session_state.kproto_step = KPROTO_STEPS[step_index + 1]

# Render current step
step = st.session_state.kproto_step
if step == "Upload":
    show_upload_ui(step_index)
elif step == "Assign Column Types":
    show_column_typing_ui(step_index)
elif step == "Run Clustering":
    show_clustering_ui(step_index)
elif step == "Review & Download":
    show_postprocessing_ui(step_index)

with st.sidebar:
    st.progress(value = (step_index + 1) / len(KPROTO_STEPS), text=st.session_state.kproto_step)