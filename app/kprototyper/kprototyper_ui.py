import streamlit as st
from kprototyper.session import KPrototyperSession
from kprototyper.logic import load_uploaded_file, preprocess_loaded_file, run_kprototypes_clustering, plot_kprototypes_results, create_session_from_df 
import uuid
import pandas as pd
from plotly.subplots import make_subplots
import io
import zipfile


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
# Step UI Utilities
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


def reset_column_assignment_flag():
    st.session_state.kproto_step_success["Assign Column Types"] = False


# -------------------------------
# Step UI Functions
# -------------------------------

def show_upload_ui(step_index):
    navigation_bar(step_index)

    # Initialize flags only once
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # Handle file upload only if not already successful
    if uploaded_file and not st.session_state.upload_success:
        try:
            df = load_uploaded_file(uploaded_file)
            session_id = f"run_{uuid.uuid4().hex[:6]}"
            session = create_session_from_df(df, session_id=session_id)
            st.session_state.kproto_sessions.append(session)

            # Save session and progress flags
            st.session_state.upload_success = True
            st.session_state.kproto_step_success["Upload"] = True
            st.session_state.kproto_last_session_id = session_id  # to retrieve later

            st.rerun()

        except ValueError as e:
            st.error(str(e))

    # Show success message and preview if upload completed
    if st.session_state.upload_success:
        st.success("File validated successfully.")

        # Get the last session (or retrieve from session_state id)
        last_session = st.session_state.kproto_sessions[-1]
        st.dataframe(last_session.input_data.head())

        st.success(f"Session '{st.session_state.kproto_last_session_id}' created with inferred column types.")


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

    if st.button("Save Assignments", icon=":material/save:", use_container_width=True, type="primary"):
        session.reset_clustering()
        st.session_state.kproto_step_success["Run Clustering"] = False

        st.success(
            f"Saved: {sum(v == 'numerical' for v in session.column_types.values())} numerical "
            f"and {sum(v == 'categorical' for v in session.column_types.values())} categorical columns."
        )
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
                disabled=(col == "name"),
                on_change=reset_column_assignment_flag
            )


def show_clustering_ui(step_index):
    navigation_bar(step_index)

    # --- Setup & Checks ---
    if not st.session_state.kproto_sessions:
        st.warning("No session found. Please upload and configure data.")
        return

    session = st.session_state.kproto_sessions[-1]

    if session.input_data is None or session.column_types is None:
        st.warning("Missing input data or column types.")
        return

    if "kproto_clustering_running" not in st.session_state:
        st.session_state.kproto_clustering_running = False

    # --- Layout containers ---
    button_container = st.container()
    log_display = st.empty()

    # --- Log utility ---
    def log(msg):
        session.logger.log(msg)
        log_display.code(session.logger.get_log_text(), language="text")

    # --- CLUSTERING RUN ---
    def run_clustering_pipeline():
        session.logger.clear()
        log("🔄 Starting clustering pipeline...")

        try:
            with st.spinner("Running clustering..."):
                clustered, overview, best_k, cost_dict, sil_dict, peak_k, shoulder_k, assignments = run_kprototypes_clustering(
                    session.input_data,
                    session.column_types,
                    k_range=(2, 30),
                    log_func=log
                )

            session.set_output(
                clustered_df=clustered,
                overview_df=overview,
                k=best_k,
                cost_dict=cost_dict,
                sil_dict=sil_dict,
                assignments=assignments
            )
            session.peak_k = peak_k
            session.shoulder_k = shoulder_k

            log(f"🎉 Clustering complete. Best k = {best_k}")
            st.session_state.kproto_step_success["Run Clustering"] = True

        except Exception as e:
            log(f"❌ Clustering failed: {e}")
            st.error(f"Clustering failed: {e}")
        finally:
            st.session_state.kproto_clustering_running = False
            st.rerun()

    # --- Execute clustering if flagged ---
    if st.session_state.kproto_clustering_running:
        run_clustering_pipeline()

    # --- Top Button Area (always shown) ---
    if st.button(" Run Clustering", use_container_width=True, type="primary", disabled=st.session_state.kproto_clustering_running, icon=":material/smart_toy:"):
        st.session_state.kproto_clustering_running = True
        st.rerun()

    # --- Show Logs ---
    if session.logger.get_log_text():
        log_display.code(session.logger.get_log_text(), language="text")

    # --- Trigger function for clarity ---
    def _trigger_clustering():
        st.session_state.kproto_clustering_running = True
        st.rerun()


def show_postprocessing_ui(step_index):
    import io, zipfile
    from datetime import datetime

    navigation_bar(step_index)

    if not st.session_state.kproto_sessions:
        st.warning("No session found.")
        return

    session = st.session_state.kproto_sessions[-1]

    if not session.is_complete():
        st.warning("Clustering not yet completed.")
        return

    # --- Header Stats ---
    st.markdown("### Clustering Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak k (max silhouette)", session.peak_k)
    col2.metric("Shoulder k (≥ 0.5 silhouette)", session.shoulder_k)
    col3.metric("Least cost k", session.selected_k)

    # --- Plot Evaluation ---
    fig = plot_kprototypes_results(
        k_range=session.cost_per_k.keys(),
        costs=session.cost_per_k,
        silhouettes=session.silhouette_per_k,
        peak_k=session.peak_k,
        shoulder_k=session.shoulder_k
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Select alternative k (optional) ---
    st.markdown("### Customize k")
    st.markdown("Allows specification of custom k value. Set by default to peak k.")
    available_ks = sorted(session.cost_per_k.keys())
    suggested_k = session.peak_k or session.selected_k
    selected_k = st.slider(
        "Select k",
        min_value=min(available_ks),
        max_value=max(available_ks),
        value=suggested_k,
        key="custom_k_selection"
    )

    # --- Regenerate data based on selected k ---
    cluster_labels = session.get_assignments_for_k(selected_k)
    df_clustered = session.input_data.copy()
    df_clustered["cluster"] = cluster_labels
    df_summary = df_clustered.groupby("cluster").agg(lambda x: x.mode().iloc[0] if not x.isnull().all() else None).reset_index()


    # --- ZIP Download ---
    st.markdown("### Download Cluster Results")
    st.markdown("Download cluster results for selected k.")

    col1, col2 = st.columns([3, 2])  # Adjust width ratio as needed

    with col1:
        format_choice = st.segmented_control(
            label="Download Format",
            options=["CSV", "XLSX"],
            label_visibility="collapsed",
            key="download_format",
            selection_mode="single",
            default="XLSX"
        )


    # --- ZIP Download ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        if format_choice == "CSV":
            zipf.writestr(f"cluster_overview_{timestamp}.csv", df_summary.to_csv(index=False))
            zipf.writestr(f"clustered_data_{timestamp}.csv", df_clustered.to_csv(index=False))
            zipf.writestr(f"original_data_{timestamp}.csv", session.input_data.to_csv(index=False))
        else:
            with io.BytesIO() as xlsx_buf:
                with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                    df_summary.to_excel(writer, sheet_name="Cluster Overview", index=False)
                    df_clustered.to_excel(writer, sheet_name="Clustered Data", index=False)
                    session.input_data.to_excel(writer, sheet_name="Original Data", index=False)
                zipf.writestr(f"clustering_output_{timestamp}.xlsx", xlsx_buf.getvalue())
    
    with col2:
        st.download_button(
            label="Download ZIP",
            data=buffer.getvalue(),
            file_name=f"clustering_results_{timestamp}.zip",
            mime="application/zip",
            use_container_width=True,
            type="secondary",
            icon=":material/download:"
        )

    
    # --- Finalize session ---
    st.markdown("### Finalize Session")
    st.markdown("Ends session. Finalized session data for the selected k is stored for the Archetyper wizard and remains availible to download from Your Data.")

    if st.button("Finalize Session", icon=":material/check_circle:", type="primary", use_container_width=True):
        # 1. Save selected k
        session.selected_k = selected_k
        session.clustered_data = df_clustered
        session.cluster_overview = df_summary
        session.status = "finalized"
        
        # 2. Confirm + cleanup + navigate
        st.success(f"Session '{session.name}' finalized and saved.")
        for key in list(st.session_state.keys()):
            if key.startswith("kproto_") or key in ["upload_success", "column_assignment_success", "clustering_success"]:
                del st.session_state[key]
        st.switch_page("home_ui.py")


# -------------------------------
# Layout and Navigation
# -------------------------------

st.markdown(f"## K-Prototyper")
step_index = KPROTO_STEPS.index(st.session_state.kproto_step)


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


# Sidebar 

with st.sidebar:
    st.markdown("### K-Prototyper Progress")

    total_steps = len(KPROTO_STEPS)
    completed_steps = sum(
        st.session_state.kproto_step_success.get(step, False) for step in KPROTO_STEPS
    )
    current_index = KPROTO_STEPS.index(st.session_state.kproto_step)

    # Progress bar reflects % of completed steps
    st.progress(completed_steps / total_steps, text=f"{completed_steps} of {total_steps} steps completed")

    # Visual step list with emoji indicators
    for i, step in enumerate(KPROTO_STEPS):
        if st.session_state.kproto_step == step:
            st.markdown(f":material/arrow_forward: **{step}**")
        elif st.session_state.kproto_step_success.get(step):
            st.markdown(f":material/check_box: {step}")
        else:
            st.markdown(f":material/check_box_outline_blank: {step}")

    # Divider + Session Info
    st.divider()
    if "kproto_sessions" in st.session_state and st.session_state.kproto_sessions:
        current_session = st.session_state.kproto_sessions[-1]
        st.caption("**Session ID:**")
        st.markdown(f":material/folder_open: `{current_session.name}`")
