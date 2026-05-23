import os
import logging
import pandas as pd
import streamlit as st
from google import genai
from google.genai.errors import APIError

# ==============================================================================
# 1. CORE SYSTEM LOGGING ARCHITECTURE (SAFE DIRECTORY VARIANT)
# ==============================================================================
LOG_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOG_DIR, "foodmetrix_system.log")

try:
    # Safely construct the directory path if it does not exist yet
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
except Exception as e:
    st.error(f"❌ System IO Initialization Failure: Could not provision logging environment schema ({str(e)}).")
    st.stop()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler() # Continues to stream to your VS Code Terminal panel
    ]
)
logger = logging.getLogger("FoodMetriX_Platform")
logger.info("Logging infrastructure safely routed to distinct local file destination.")

# ==============================================================================
# 2. STREAMLIT CONFIGURATION & INTERFACE THEME
# ==============================================================================
st.set_page_config(
    page_title="FoodMetriX.AI - Industrial QA & Compliance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent session states for vector matrix simulations
if "embedded_assets" not in st.session_state:
    st.session_state.embedded_assets = []
if "tenant_context" not in st.session_state:
    st.session_state.tenant_context = "FoodCorp_Plant_A"

# ==============================================================================
# 3. SECURE BACKEND AUTHENTICATION INTERCEPTOR
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_gemini_client():
    """
    Safely extracts credentials from environment secrets and provisions the SDK client.
    """
    try:
        logger.info("Intercepting environment space for secure API keys...")
        if not st.secrets or "GEMINI_API_KEY" not in st.secrets:
            raise KeyError("Identifier 'GEMINI_API_KEY' was not found inside local .streamlit/secrets.toml configurations.")
        
        api_key_string = st.secrets["GEMINI_API_KEY"]
        if not api_key_string or api_key_string.strip() == "":
            raise ValueError("The GEMINI_API_KEY string variable payload is blank or empty.")
            
        client = genai.Client(api_key=api_key_string)
        logger.info("Google Gemini SDK client context bound successfully to local runtime.")
        return client
        
    except KeyError as ke:
        logger.critical(f"Security Core Halt: Configuration mapping failed. Detail: {str(ke)}")
        st.error("🔒 Security Halt: App config environment parameters missing. Check local system files.")
        st.stop()
    except ValueError as ve:
        logger.critical(f"Security Core Halt: Bad variable assignment. Detail: {str(ve)}")
        st.error("🔒 Security Halt: Configured key formatting error detected. Operations suspended.")
        st.stop()
    except Exception as e:
        logger.critical(f"Infrastructure Panic: Runtime binding crash: {str(e)}", exc_info=True)
        st.error(f"❌ Core System Crash: Internal initialization failed ({type(e).__name__}).")
        st.stop()

# Instantiate the globally isolated client
client = initialize_gemini_client()

# ==============================================================================
# 4. DEFENSIVE INGESTION PIPELINE FUNCTIONS
# ==============================================================================
def defensive_parse_document(uploaded_file):
    """
    Executes deep syntax parsing and matrix conversion while catching formatting corruptions.
    """
    logger.info(f"Target file pipeline engagement triggered: {uploaded_file.name}")
    
    try:
        if uploaded_file.size == 0:
            raise ValueError("File content payload validation failed: File size registers as 0 bytes.")
            
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        extracted_elements_count = 0
        
        # Branch evaluation routing by content types
        if file_extension in ['.xlsx', '.xls']:
            logger.info("Engaging openpyxl compilation engine matrix rows...")
            # Read into dataframe to validate data layout structures
            df = pd.read_excel(uploaded_file)
            extracted_elements_count = len(df)
            logger.info(f"DataFrame ingestion verified. Loaded shape metrics: {df.shape}")
            
        elif file_extension == '.pdf':
            logger.info("Engaging pdfplumber extraction node...")
            import pdfplumber
            with pdfplumber.open(uploaded_file) as pdf:
                if not pdf.pages:
                    raise ValueError("Target PDF document layout contains zero active readable vector pages.")
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_elements_count += len(text.split('\n'))
            logger.info(f"PDF content string parse complete. Extracted lines metric: {extracted_elements_count}")
            
        elif file_extension == '.csv':
            logger.info("Engaging engine pandas parser routine...")
            df = pd.read_csv(uploaded_file)
            extracted_elements_count = len(df)
            
        else:
            raise TypeError(f"Ingested format identifier '{file_extension}' breaks site standard whitelist specs.")
            
        if extracted_elements_count == 0:
            raise ValueError("The pipeline parsed the document framework, but discovered no readable row/text elements.")
            
        return {
            "success": True,
            "filename": uploaded_file.name,
            "elements": extracted_elements_count
        }
        
    except TypeError as te:
        logger.warning(f"File ingestion rejected due to signature mismatch: {str(te)}")
        st.warning(f"⚠️ Validation Failure: {str(te)}")
        return {"success": False, "reason": str(te)}
        
    except ValueError as ve:
        logger.warning(f"Data constraint anomaly surfaced during file execution: {str(ve)}")
        st.error(f"📁 Ingestion Fault: Unable to map file payload. {str(ve)}")
        return {"success": False, "reason": str(ve)}
        
    except Exception as e:
        logger.error(f"Critical data block translation breakdown: {str(e)}", exc_info=True)
        st.error(f"❌ Processing Interrupted: Structural parse failure inside file. ({type(e).__name__})")
        return {"success": False, "reason": type(e).__name__}

# ==============================================================================
# 5. SAFE AI INFERENCE QUERY ROUTER
# ==============================================================================
def execute_safe_inference(user_prompt, data_context=""):
    """
    Forwards user compliance inquiries to Gemini endpoints behind defensive error handlers.
    """
    logger.info("Marshalling contextual prompt payloads for deployment to gemini-2.5-flash...")
    
    system_instruction = """
    You are an advanced Industrial Quality Assurance & Compliance Audit AI for FoodMetrix.AI.
    Your objective is analyzing factory floor datasets, asset logs, and verification reports.
    Always anchor analyses to the parameters and datasets displayed inside 'Current Embedded Factory Files'.
    If facts cannot be parsed out of context fields, return: 'Data context missing for this metric.'
    """
    
    structured_contents = f"""
    Current Embedded Factory Files Context:
    \"\"\"
    {data_context if data_context else 'No data assets loaded.'}
    \"\"\"
    
    User Infiltration Query: {user_prompt}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=structured_contents,
            config={"system_instruction": system_instruction, "temperature": 0.15}
        )
        logger.info("Inference streaming resolved successfully without token drops.")
        return response.text
        
    except APIError as ae:
        logger.error(f"Upstream Google Gemini Service Error: Code {ae.code} - {ae.message}", exc_info=True)
        st.error("🤖 Engine Delay: Cloud processing nodes are struggling to route queries. Retrying pipeline...")
        return None
    except Exception as e:
        logger.error(f"Local request serialization breakdown: {str(e)}", exc_info=True)
        st.error("❌ Network Endpoint Error: Failed to execute generative analytical routing layer.")
        return None

# ==============================================================================
# 6. STREAMLIT INTERFACE COMPOSER LAYOUT (SIDEBAR & MAINFRAME)
# ==============================================================================

# ----------------- SIDEBAR INTERACTIVE DASHBOARD -----------------
with st.sidebar:
    st.markdown("### 🏢 Tenant Identity Control")
    selected_tenant = st.selectbox(
        "Active Organization Context:",
        ["FoodCorp_Plant_A", "FoodCorp_Plant_B", "Logistics_Hub_Melbourne"],
        key="tenant_selector"
    )
    st.session_state.tenant_context = selected_tenant
    
    st.markdown("---")
    st.markdown("### 📥 Data Loading Terminal")
    st.caption("Ingest production line assets:")
    
    uploaded_file = st.file_uploader(
        "Drop records panel:",
        type=["csv", "xlsx", "xls", "pdf"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 Target Loaded: {uploaded_file.name}")
        if st.button("Process & Vectorize Document Asset", type="primary", use_container_width=True):
            with st.spinner("Compiling matrix vectors... Please hold..."):
                result = defensive_parse_document(uploaded_file)
                if result["success"]:
                    st.session_state.embedded_assets.append({
                        "name": result["filename"],
                        "elements": result["elements"]
                    })
                    st.toast("⚡ Document compiled into safe vector arrays!", icon="✅")
                    logger.info(f"Successfully integrated tracking arrays for {result['filename']}")

# ----------------- CENTRAL APPLICATION VIEWPORT -----------------
st.markdown("# 🎯 FoodMetrix.AI")
st.markdown("### Industrial Intelligence Platform for Predictive Quality Assurance & Compliance Audit Analytics")
st.caption(f"🔒 Secured Enterprise Tenant Workspace Environment | Scope: `{st.session_state.tenant_context}` | Powered by Gemini-2.5-Flash")

# Visual segmentation tab selectors
tab_storage, tab_agent = st.tabs(["📊 Vector Storage Profile", "💬 Contextual AI Agent"])

# --- TAB 1: KNOWLEDGE STORAGE DATABASE MANAGEMENT ---
with tab_storage:
    st.markdown("## Current Embedded Factory Files")
    
    if not st.session_state.embedded_assets:
        st.info("📂 No active knowledge assets discovered for this tenant context database.")
    else:
        for idx, asset in enumerate(st.session_state.embedded_assets):
            with st.container(border=True):
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.markdown(f"**📝 LINKED_ASSET →** `{asset['name']}`")
                with col2:
                    st.markdown(f"🔢 Vector Elements: `{asset['elements']}` rows/lines")
                with col3:
                    if st.button("Purge", key=f"purge_{idx}", use_container_width=True):
                        logger.info(f"Purging tracked instance memory for array element: {asset['name']}")
                        st.session_state.embedded_assets.pop(idx)
                        st.rerun()

# --- TAB 2: ACTIVE CONTEXT ANALYTICS CONSOLE ---
with tab_agent:
    st.markdown("## Live Operational Intelligence Console")
    st.caption("Interrogate active plant datasets to cross-reference performance matrices against QA specifications.")
    
    # Simple isolated session loop storage for user conversations
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    user_query = st.chat_input("Enter compliance query parameter (e.g., 'Identify anomalies in batch 26 line logs')")
    
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Build contextual footprint string from current state variables
        mock_context_footprint = ""
        if st.session_state.embedded_assets:
            mock_context_footprint = f"Active Assets Tracking References: {str(st.session_state.embedded_assets)}"
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing operational signals..."):
                agent_reply = execute_safe_inference(user_query, data_context=mock_context_footprint)
                if agent_reply:
                    st.markdown(agent_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": agent_reply})