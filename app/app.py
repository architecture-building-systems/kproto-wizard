import streamlit as st

pages = {
    "CEA Database Wizard": [
        st.Page("home_ui.py", title="Home", icon=":material/home:"),
        st.Page("kprototyper/kprototyper_ui.py", title="K-Prototyper", icon=":material/smart_toy:"),
        st.Page("archetyper/archetyper_ui.py", title="Archetyper", icon=":material/villa:"),
        st.Page("userdata_ui.py", title="Your Data", icon=":material/folder:")
    ],
    "Reference": [
        st.Page("resources_ui.py", title="Resources"),
        st.Page("about_ui.py", title="About")
    ]
}

if "page_id" not in st.session_state:
    st.session_state.page_id = "home_page.py"

pg = st.navigation(pages, position="sidebar")
pg.run()