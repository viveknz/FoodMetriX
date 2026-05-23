# app.py
import streamlit as st
from src.config import logger, get_gemini_client
from src.parsers import parse_to_dataframe, parse_pdf_text
from src.brain import compute_and_store_embeddings, search_similarity

# Initialize application settings & core client
st.set_page_config(page_title="FoodMetriX.AI", page_icon="🎯", layout="wide")
client = get_gemini_client()

# Session state checks
if "embedded_assets" not in st.session_state: st.session_state.embedded_assets = []
if "vector_store" not in st.session_state: st.session_state.vector_store = {}

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 🏢 Tenant Identity Control")
    tenant = st.selectbox("Active Organization:", ["FoodCorp_Plant_A", "FoodCorp_Plant_B"])
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📥 Data Ingestion Panel:", type=["csv", "xlsx", "pdf"])
    
    if uploaded_file and st.button("Process & Vectorize Document Asset", type="primary", use_container_width=True):
        with st.spinner("Compiling matrix vectors..."):
            try:
                if uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
                    df = parse_to_dataframe(uploaded_file)
                    cnt = compute_and_store_embeddings(df, uploaded_file.name, tenant, is_tabular=True)
                elif uploaded_file.name.endswith('.pdf'):
                    chunks = parse_pdf_text(uploaded_file)
                    cnt = compute_and_store_embeddings(chunks, uploaded_file.name, tenant, is_tabular=False)
                
                st.session_state.embedded_assets.append({"name": uploaded_file.name, "elements": cnt, "tenant": tenant})
                st.toast("⚡ Data successfully vectorized!", icon="✅")
            except Exception as e:
                st.error(f"Ingestion crashed: {str(e)}")

# --- MAIN PAGE DISPLAY ---
st.markdown("# 🎯 FoodMetrix.AI")
tab_storage, tab_agent = st.tabs(["📊 Vector Storage Profile", "💬 Contextual AI Agent Console"])

# Render layout frames by pulling logic out of src package safely...
# (UI logic for rendering lists and handling st.chat_input goes here seamlessly)