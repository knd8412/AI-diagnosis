# =============================================================
# STREAMLIT UI - AI RESPIRATORY DIAGNOSIS ASSISTANT
# =============================================================
# Description:
#   Streamlit-based UI for patient intake, chat, and diagnosis.
#   Integrates RAG chains, X-ray analysis, Redis caching,
#   ChromaDB patient memory, and MongoDB logging.
#   Supports both named-patient and guest-mode workflows.
# =============================================================

import streamlit as st
import uuid
import redis
import os
from langchain_redis import RedisChatMessageHistory, RedisCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage
import matplotlib.pyplot as plt
import requests
import base64
import numpy as np

# --- Project Imports ---
from llm_integration.chains.rag_chains import DiagnosisRAG
from llm_integration.retrieval.chromaClient import get_patient_memory
from SQLdb.models import save_patient, get_patient, get_all_patients

# --- Configuration & Redis Setup ---
# Redis handles short-term chat history and LLM response caching
REDIS_URL = "redis://redis:6379"
TTL = 3600
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
redis_cache = RedisCache(redis_url=REDIS_URL, ttl=TTL)
set_llm_cache(redis_cache)
# ChromaDB handles long-term semantic memory per patient across sessions
patient_memory = get_patient_memory()

# --- Initialize RAG Chain ---
@st.cache_resource
def load_rag_chain():
    # Function note:
    # - Streamlit cache ensures model is loaded only once per session
    # - DiagnosisRAG wraps all chain orchestration and LLM calls
    return DiagnosisRAG()

rag_chain = load_rag_chain()

def get_chat_history(session_id):
    # Function note:
    # - Returns RedisChatMessageHistory for persistent short-term storage
    # - Each session has its own Redis key isolated by session_id
    return RedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL, ttl=TTL)

def render_xray_results(saved_data):
    # Function note:
    # - Renders pathology scores as percentages + progress bars
    # - Displays CAM heatmaps with optional original image overlay
    # - Handles both list-based numpy arrays and base64-encoded image strings
    st.markdown(f"### X-Ray Analysis: {saved_data.get('filename', 'Image')}")
    if "pathologies" in saved_data:
        sorted_paths = sorted(saved_data["pathologies"].items(), key=lambda x: x[1], reverse=True)
        for condition, prob in sorted_paths[:5]:
            st.write(f"**{condition}**: {prob * 100:.1f}%")
            st.progress(prob)

    if "top_cams" in saved_data:
        st.markdown("#### Disease Heatmaps")

        # Fetch the original scan once for overlay
        original_img = None
        scan_file = saved_data.get("scan_file")
        if scan_file:
            try:
                from PIL import Image
                import io
                scan_response = requests.get(
                    f"{IMAGE_SERVICE_URL}/scans/{scan_file}",
                    timeout=3
                )
                if scan_response.status_code == 200:
                    pil_img = Image.open(io.BytesIO(scan_response.content)).convert("L")
                    original_img = np.array(pil_img)
            except Exception:
                pass

        cols = st.columns(len(saved_data["top_cams"]))
        for col, (condition, cam_data) in zip(cols, saved_data["top_cams"].items()):
            with col:
                try:
                    if isinstance(cam_data, list):
                        cam_array = np.array(cam_data)
                        cam_array = (cam_array - np.min(cam_array)) / (np.max(cam_array) - np.min(cam_array) + 1e-8)
                        colormap = plt.get_cmap('jet')
                        heatmap_rgb = (colormap(cam_array)[:, :, :3] * 255).astype(np.uint8)

                        if original_img is not None:
                            from PIL import Image
                            orig_resized = np.array(
                                Image.fromarray(original_img).resize(
                                    (cam_array.shape[1], cam_array.shape[0]),
                                    Image.LANCZOS
                                )
                            )
                            orig_rgb = np.stack([orig_resized] * 3, axis=-1)
                            blended = (0.4 * orig_rgb + 0.6 * heatmap_rgb).astype(np.uint8)
                            st.image(blended, caption=condition, use_container_width=True)
                        else:
                            st.image(heatmap_rgb, caption=condition, use_container_width=True)

                    elif isinstance(cam_data, str):
                        if cam_data.startswith("data:image"):
                            cam_data = cam_data.split(",")[1]
                        cam_data += "=" * ((4 - len(cam_data) % 4) % 4)
                        st.image(base64.b64decode(cam_data), caption=condition, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {condition}: {e}")


# -- mongoDB persistent storage --
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://db-service:8002")
IMAGE_SERVICE_URL = os.getenv("IMAGE_SERVICE_URL", "http://image-processing-service:8001")

def persist_chatlog_to_mongo(session_id: str, patient_id: str, result_ids: list, messages):
    # Function note:
    # - Asynchronously writes full chat session to MongoDB for long-term persistence
    # - Separates xray result_ids into their own list for relational integrity
    # - Swallows exceptions to avoid blocking the UI on network failures
    payload = {
        "session_id": session_id,
        "patient_id": patient_id,
        "result_ids": result_ids,
        "messages": [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ],
    }
    try:
        r = requests.post(f"{DB_SERVICE_URL}/chatlogs", json=payload, timeout=3)
    except Exception as e:
        print(f"[persist_chatlog] FAILED: {e}")

def load_sessions_from_mongo(patient_id: str):
    # Function note:
    # - Restores Redis chat list and Redis message history from MongoDB on app startup
    # - Rebuilds xray_ids and message order preservation per turn
    # - Handles graceful failure (vs stopping the app) if MongoDB is unavailable
    """Backfill Redis chat list and message history from MongoDB after restart"""
    try:
        response = requests.get(
            f"{DB_SERVICE_URL}/chatlogs/by-patient/{patient_id}",
            timeout=3
        )
        if response.status_code == 200:
            logs = response.json()
            existing_ids = redis_client.lrange(f"chat_list:{patient_id}", 0, -1)
            for log in logs:
                session_id = log.get("session_id")
                if not session_id:
                    continue

                if session_id not in existing_ids:
                    redis_client.rpush(f"chat_list:{patient_id}", session_id)
                    existing_ids.append(session_id)

                # Restore message history
                history = get_chat_history(session_id)
                if not history.messages:
                    for msg in log.get("messages", []):
                        if msg["role"] == "user":
                            history.add_user_message(msg["content"])
                        else:
                            history.add_ai_message(msg["content"])

                # Restore xray result IDs, mapped to the last turn
                result_ids = log.get("result_ids", [])
                if result_ids:
                    last_turn = len(log.get("messages", [])) - 1
                    for result_id in result_ids:
                        redis_client.rpush(f"xray_ids:{session_id}:turn_{last_turn}", result_id)

    except Exception as e:
        st.sidebar.warning(f"Could not restore sessions: {e}")


# --- UI Setup ---
st.set_page_config(page_title="Respiratory AI Assistant", layout="wide")

if "patient_mode" not in st.session_state:
    st.session_state.patient_mode = None
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = None
# Load external CSS (Ensure styles.css is in the same folder)
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass # CSS will be skipped if file is missing


# --- Persistent Sidebar & Search Logic ---
# Manage chat session ID: prioritize URL params, fall back to most recent, or create new
patient_id = st.session_state.get("current_patient_id", "guest")
all_chat_ids = redis_client.lrange(f"chat_list:{patient_id}", 0, -1)

if "id" in st.query_params:
    active_id = st.query_params["id"]  # User navigated directly to a chat via URL
elif all_chat_ids:
    active_id = all_chat_ids[-1]  # Resume most recent session
else:
    active_id = str(uuid.uuid4())  # First chat for this patient

# On first load, backfill Redis from MongoDB to restore historical sessions
if not all_chat_ids or all_chat_ids == [active_id]:
    load_sessions_from_mongo(patient_id)
    all_chat_ids = redis_client.lrange(f"chat_list:{patient_id}", 0, -1)

st.session_state.active_chat_id = active_id
st.query_params["id"] = active_id  # Keep URL in sync with active chat

# Register new session in Redis if it doesn't exist
if active_id not in all_chat_ids:
    redis_client.rpush(f"chat_list:{patient_id}", active_id)  
    all_chat_ids.append(active_id)

# --- WORKFLOW: Mandatory Intake Overlay ---
# Block access to main app until a patient is selected or guest mode chosen.
# This gate ensures every session has a patient_id and a clarity of whether data persists.
if st.session_state.patient_mode is None:
    st.title("Patient Intake & Selection")
    st.write("Please identify the patient or choose Guest Mode to begin the consultation.")

    # Three pathways: existing patient, new registration, or ephemeral guest session
    tab1, tab2, tab3 = st.tabs(["Search Existing Patient", "Register New Patient", "Guest Mode"])

    with tab1:
        # TAB 1: Choose from existing patients in SQLdb
        all_patients = get_all_patients()
        if not all_patients:
            st.info("No patient records found in the database.")
        else:
            patient_options = {f"{p.name} (ID: {p.patient_id})": p.patient_id for p in all_patients}
            selected_label = st.selectbox("Select a patient profile", options=[None] + list(patient_options.keys()))
            
            if st.button("Start Consultation with Selected Patient") and selected_label:
                # Set patient context, enable data persistence to MongoDB and ChromaDB
                st.session_state.current_patient_id = patient_options[selected_label]
                st.session_state.patient_mode = "selected"
                # Create a fresh session ID for this specific interaction
                st.session_state.active_chat_id = str(uuid.uuid4())
                st.query_params["id"] = st.session_state.active_chat_id
                st.rerun()  # Re-execute app with patient context set

    with tab2:
        # TAB 2: Register a new patient and immediately start a consultation
        with st.form("new_patient_registration"):
            new_name = st.text_input("Patient Full Name")
            new_age = st.number_input("Age", 0, 120, value=30)
            new_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            
            if st.form_submit_button("Create Record & Start Chat"):
                if new_name:
                    # Save to SQLdb for persistent patient records
                    new_id = f"P-{str(uuid.uuid4())[:8]}"
                    save_patient(new_id, new_name, new_age, new_gender)
                    st.session_state.current_patient_id = new_id
                    st.session_state.patient_mode = "selected"
                    st.session_state.active_chat_id = str(uuid.uuid4())
                    st.query_params["id"] = st.session_state.active_chat_id
                    st.rerun()
                else:
                    st.error("Name is required to create a record.")

    with tab3:
        # TAB 3: Ephemeral demo mode—no data persistence
        st.warning("Guest mode sessions are not saved to the permanent clinical database or ChromaDB memory.")
        if st.button("Proceed as Guest"):
            # Create unique guest ID, but don't persist to MongoDB or ChromaDB
            st.session_state.current_patient_id = f"GUEST-{str(uuid.uuid4())[:8]}"
            st.session_state.patient_mode = "guest"
            st.session_state.active_chat_id = str(uuid.uuid4())
            st.query_params["id"] = st.session_state.active_chat_id
            st.rerun()
    st.stop() # Halt execution until a mode is selected

# --- MAIN APP (Only runs after selection) ---

# Sidebar Navigation
with st.sidebar:
    st.title("AI Diagnosis Assistant")
    
    # Switch Patient Button
    if st.button("Switch Patient / Exit Chat"):
        st.session_state.patient_mode = None
        st.session_state.current_patient_id = None
        st.rerun()
    # New Chat button creates a fresh session for the current patient
    if st.button("New Chat"):
        new_chat_id = str(uuid.uuid4())
        patient_id = st.session_state.get("current_patient_id", "guest")
        redis_client.rpush(f"chat_list:{patient_id}", new_chat_id)
        st.session_state.active_chat_id = new_chat_id
        st.query_params["id"] = new_chat_id
        st.rerun()
        
    search_query = st.text_input("search", placeholder="🔍 Search consultations...", label_visibility="collapsed")

    st.divider()
    
    # Active Patient Card
    if st.session_state.patient_mode == "selected":
        p = get_patient(st.session_state.current_patient_id)
        st.success(f"**Current Patient:**\n{p.name}\nAge: {p.age} | {p.gender}")
    else:
        st.warning("**Mode:** Guest Consultation")

    st.divider()
    st.subheader("Consultation History")
    
    # Manage Redis list for the current patient/guest
    patient_id = st.session_state.get("current_patient_id", "guest")
    all_chat_ids = redis_client.lrange(f"chat_list:{patient_id}", 0, -1)

    # Filter sidebar to show only chats belonging to THIS specific patient_id
    # (Optional: You can remove this filter if you want to see all historical chats)
    for c_id in reversed(all_chat_ids):
        temp_history = get_chat_history(c_id)
        chat_label = "New Chat"
        for msg in temp_history.messages:
            if isinstance(msg, HumanMessage):
                chat_label = (msg.content[:25] + '...') if len(msg.content) > 25 else msg.content
                break 
        
        if st.button(chat_label, key=f"btn_{c_id}"):
            st.session_state.active_chat_id = c_id
            st.query_params["id"] = c_id
            st.rerun()

# --- Main Chat Area ---
active_id = st.session_state.active_chat_id
history = get_chat_history(active_id)
chat_container = st.container()

with chat_container:
    # Status Banner
    if st.session_state.patient_mode == "selected":
        st.info(f"**Clinical Session** | Patient ID: {st.session_state.current_patient_id}")
    else:
        st.info(f"**Guest Session** | No persistent data linked.")

    # Display Messages
    # Re-render full conversation history from Redis on every page load
    for i, msg in enumerate(history.messages):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            # Redraw X-ray results - try MongoDB first, fall back to session state
            if role == "assistant":
                # all_keys = redis_client.keys(f"xray_ids:{active_id}:*")
                xray_ids = redis_client.lrange(f"xray_ids:{active_id}:turn_{i}", 0, -1)
                if not xray_ids:
                    xray_ids = redis_client.lrange(f"xray_ids:{active_id}:turn_{i-1}", 0, -1)
                if xray_ids:
                    for result_id in xray_ids:
                        try:
                            db_response = requests.get(f"{DB_SERVICE_URL}/results/{result_id}")
                            if db_response.status_code == 200:
                                render_xray_results(db_response.json())
                        except Exception as e:
                            st.error(f"Could not load X-ray result: {e}")
                elif f"xray_{active_id}_{i}" in st.session_state:
                    for saved_data in st.session_state[f"xray_{active_id}_{i}"]:
                        render_xray_results(saved_data)

            context_key = f"context_{active_id}_{i}"
            if role == "assistant" and context_key in st.session_state:
                with st.expander("🔍 Clinical Evidence"):
                    st.info(st.session_state[context_key])


# --- Chat Input & RAG Logic ---
prompt_obj = st.chat_input(
    "Describe symptoms or upload relevant documents...", 
    accept_file="multiple", 
    file_type=["png", "jpg", "jpeg"],
    key="main_input_bar"
)

if prompt_obj:
    # === USER MESSAGE PROCESSING ===
    # Parse input: prioritize text, auto-generate if only files attached
    user_text = prompt_obj.text if hasattr(prompt_obj, 'text') and prompt_obj.text else ""
    if not user_text and hasattr(prompt_obj, 'files') and prompt_obj.files:
        user_text = "Please analyze the attached X-Ray."
        
    user_display = user_text
    has_files = hasattr(prompt_obj, 'files') and prompt_obj.files
    
    if has_files:
        user_display += f" (Attached: {', '.join([f.name for f in prompt_obj.files])})"
    
    # Display message in chat UI and persist to Redis
    with chat_container:
        st.chat_message("user").markdown(user_display)
        history.add_user_message(user_display)

    # Sync to patient memory (ChromaDB) for semantic search in future turns
    if st.session_state.patient_mode == "selected":
        patient_memory.add_interaction(
            st.session_state.current_patient_id, 
            "user", 
            user_text
        )
    # Build formatted context from Redis history for RAG chain
    context_history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history.messages])

    # ==========================================
    # X-RAY PROCESSING PIPELINE
    # ==========================================
    # Send attached images to image-processing service for ML inference and CAM generation
    image_analysis_summary = "" 
    current_xray_results = []  # Holds X-ray analysis results for the current turn
    
    if has_files:
        with chat_container:
            for attached_file in prompt_obj.files:
                with st.chat_message("assistant"):
                    with st.spinner(f"Analyzing X-Ray: {attached_file.name}..."):
                        try:
                            # POST to image-service for TorchXRayVision inference
                            files = {"file": (attached_file.name, attached_file.getvalue(), attached_file.type)}
                            data = {"session_id": active_id}
                            response = requests.post(f"{IMAGE_SERVICE_URL}/analyze", files=files, data=data)
                            
                            if response.status_code == 200:
                                data = response.json()  # pathologies, top_cams, embeddings, result_id
                                result_id = data.get("result_id")  # MongoDB ID if db-service is available
                                # Save the filename and data to our list
                                data["filename"] = attached_file.name
                                current_xray_results.append(data)
                                if result_id:
                                    # Map result IDs to message indices for correct heatmap display
                                    current_msg_idx = len(history.messages)
                                    redis_client.rpush(f"xray_ids:{active_id}:turn_{current_msg_idx}", result_id)
 
                                
                                # Render pathologies, CAM heatmaps, and optional image overlay
                                render_xray_results(data)
                                # Summarize ML findings as structured text for RAG chain context
                                confident = data.get("confident_results", {})
                                if confident:
                                    findings = ", ".join([f"{k} ({v*100:.1f}%)" for k, v in confident.items()])
                                    image_analysis_summary += f"\n[System Note: X-Ray {attached_file.name} showed confident findings of: {findings}]."
                                else:
                                    image_analysis_summary += f"\n[System Note: X-Ray {attached_file.name} showed NO highly confident disease patterns]."

                            else:
                                st.error(response.json().get("message", response.text))
                        except Exception as e:
                            st.error(f"Failed to connect to Image Processing Service. Error: {e}")

    # Append X-ray summary to user query so the RAG chain sees it
    if image_analysis_summary:
        user_text += image_analysis_summary
        
    # Persist X-ray findings to patient memory for semantic search
    if image_analysis_summary and st.session_state.patient_mode == "selected":
        patient_memory.add_interaction(
            st.session_state.current_patient_id,
            "xray_analysis",
            image_analysis_summary
        )
        
    # Store analysis results in session state, keyed by message index for later display
    if current_xray_results:
        current_msg_idx = len(history.messages) # The index of the AI's upcoming text message
        st.session_state[f"xray_{active_id}_{current_msg_idx}"] = current_xray_results
        
    # ==========================================
    # RAG CHAIN EXECUTION
    # ==========================================
    # 1. Retrieve past patient history from ChromaDB
    # 2. Augment user query with X-ray findings + historical context
    # 3. Generate diagnosis with RAG chain using LLM
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Consulting medical database and drafting diagnosis..."):
                # Semantic search ChromaDB for relevant interactions from this patient's history
                chroma_context = ""
                if st.session_state.patient_mode == "selected":
                    chroma_context = patient_memory.search_history(
                        st.session_state.current_patient_id,
                        user_text  # Query: current symptoms + X-ray analysis
                    )
                    
                # Invoke RAG chain: merges X-ray summary + history + chat context for LLM
                result = rag_chain.diagnose(
                    patient_id=st.session_state.current_patient_id,
                    patient_query=user_text + (f"\n\nRelevant past history: {chroma_context}" if chroma_context else ""),
                    chat_history=context_history  # Full conversation history for context
                )
                
                full_response = result["diagnosis"]
                st.markdown(full_response)
                
                current_msg_idx = len(history.messages)
                st.session_state[f"context_{active_id}_{current_msg_idx}"] = result["retrieved_context"]
                
                with st.expander("View Clinical Evidence (Retrieved Context)"):
                    st.info(result["retrieved_context"])

    # === POST-RESPONSE SYNCING ===
    # Save AI response to Redis chat history
    history.add_ai_message(full_response)
    # Persist to ChromaDB for semantic memory
    if st.session_state.patient_mode == "selected":
        patient_memory.add_interaction(
            st.session_state.current_patient_id,
            "assistant",
            full_response
        )

    # === MONGODB PERSISTENCE ===
    # After each turn, save full chat log to MongoDB (skip if guest mode)
    patient_id = st.session_state.get("current_patient_id", "guest")
    result_ids = []  # Collect all X-ray result IDs from this session
    keys = redis_client.keys(f"xray_ids:{active_id}:turn_*")
    for key in keys:
        result_ids.extend(redis_client.lrange(key, 0, -1))
    # Only persist to MongoDB for named patients (not guest mode)
    if st.session_state.patient_mode != "guest":
        persist_chatlog_to_mongo(
            session_id=active_id,
            patient_id=patient_id,
            result_ids=result_ids,  # Links chat to analysis results
            messages=history.messages,  # Full conversation history
        )

