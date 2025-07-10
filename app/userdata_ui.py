import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

import pandas as pd
import streamlit as st

from archetyper.session import ArchetyperSession
from kprototyper.session import KPrototyperSession  # adjust if needed

def userdata_page():
    st.title("Your Data")
    st.markdown("Active session class instances")

    # --- KPrototyperSession instances ---
    kproto_keys = _find_instances_of(KPrototyperSession)
    st.subheader(":material/smart_toy: KPrototyperSession Instances")
    if kproto_keys:
        for key in sorted(kproto_keys):
            st.markdown(f"- `{key}`")
    else:
        st.info("No KPrototyperSession instances found.")

    # --- ArchetyperSession instances ---
    arch_keys = _find_instances_of(ArchetyperSession)
    st.subheader(":material/villa: ArchetyperSession Instances")
    if arch_keys:
        for key in sorted(arch_keys):
            st.markdown(f"- `{key}`")
    else:
        st.info("No ArchetyperSession instances found.")

    if st.button("Refresh"):
        st.rerun()

# --- Helper ---
def _find_instances_of(cls):
    return [k for k, v in st.session_state.items() if isinstance(v, cls)]

userdata_page()