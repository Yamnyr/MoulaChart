import streamlit as st

st.set_page_config(page_title="📈 MoulaChart", page_icon="📊", layout="wide")

# --- Navigation principale sans sous-menus ---
pages = [
    st.Page("pages/0_Dashboard.py", title="📊 Dashboard"),
    st.Page("pages/1_Portefeuille.py", title="💼 Portefeuille"),
]

pg = st.navigation(pages, position="top")
pg.run()
