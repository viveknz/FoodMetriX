# src/config.py
import os
import logging
import streamlit as st
from google import genai

LOG_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "foodmetrix_system.log")

# Ensure logs folder structure exists safely
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging centrally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FoodMetriX_Core")

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Safely initializes and returns the Google GenAI Client."""
    if not st.secrets or "GEMINI_API_KEY" not in st.secrets:
        logger.critical("Security Core Halt: GEMINI_API_KEY identifier missing.")
        st.error("🔒 Security Halt: API parameters missing. Check configuration.")
        st.stop()
        
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])