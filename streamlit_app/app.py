import streamlit as st

st.set_page_config(
    page_title="Zebra Migration",
    page_icon="🦓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦓 Zebra Migration & Wetter")
st.markdown("Willkommen zur Zebra-Migrationsanalyse! Nutze die Seiten im Sidebar, um zwischen verschiedenen Ansichten zu wechseln.")

st.markdown("""
### Verfügbare Seiten:

- **📊 Karte** - Interaktive Kartenansicht mit Filteroptionen
- **🎬 Animation** - Tag-für-Tag Animation der Zebra-Migration

Wähle eine Seite aus dem Sidebar-Menü aus, um zu beginnen.
""")