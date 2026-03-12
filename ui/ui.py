import streamlit as st
import uuid
import redis
from langchain_redis import RedisChatMessageHistory, RedisCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage, AIMessage

# --- Project Imports ---
from llm_integration.chains.rag_chains import DiagnosisRAG

# --- Configuration & Redis Setup ---
REDIS_URL = "redis://redis:6379"
TTL = 3600
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
redis_cache = RedisCache(redis_url=REDIS_URL, ttl=TTL)
set_llm_cache(redis_cache)

# --- Initialize RAG Chain ---
@st.cache_resource
def load_rag_chain():
    return DiagnosisRAG()

rag_chain = load_rag_chain()

def get_chat_history(session_id):
    return RedisChatMessageHistory(session_id=session_id, redis_url=REDIS_URL, ttl=TTL)

# --- UI Setup ---
st.set_page_config(page_title="Respiratory AI Assistant", layout="wide")

# Load external CSS (Ensure styles.css is in the same folder)
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass # CSS will be skipped if file is missing

# --- Persistent Sidebar & Search Logic ---
all_chat_ids = redis_client.lrange("chat_list", 0, -1)

if "id" in st.query_params:
    active_id = st.query_params["id"]
elif all_chat_ids:
    active_id = all_chat_ids[-1]
else:
    active_id = str(uuid.uuid4())

st.session_state.active_chat_id = active_id
st.query_params["id"] = active_id

if active_id not in all_chat_ids:
    redis_client.rpush("chat_list", active_id)
    all_chat_ids.append(active_id)

with st.sidebar:
    st.title("AI Diagnosis Assistant")
    
    if st.button("➕ New Consultation"):
        new_id = str(uuid.uuid4())
        st.query_params["id"] = new_id
        st.session_state.active_chat_id = new_id
        st.rerun()

    search_query = st.text_input("search", placeholder="🔍 Search consultations...", label_visibility="collapsed")
    st.divider()

    st.subheader("Previous Consultations")
    for c_id in reversed(all_chat_ids):
        temp_history = get_chat_history(c_id)
        chat_label = "New Consultation"
        for msg in temp_history.messages:
            if isinstance(msg, HumanMessage):
                chat_label = (msg.content[:25] + '...') if len(msg.content) > 25 else msg.content
                break 
        
        if st.button(chat_label, key=c_id):
            st.query_params["id"] = c_id
            st.session_state.active_chat_id = c_id
            st.rerun()

# --- Main Chat Area ---
active_id = st.session_state.active_chat_id
history = get_chat_history(active_id)
chat_container = st.container()

with chat_container:
    if len(history.messages) == 0:
        st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
        st.markdown("<h1 class='greeting-text'>How can I help you?</h1>", unsafe_allow_html=True)
    else:
        for i, msg in enumerate(history.messages):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)
                
                # PERSISTENCE FIX: Show context expander if it exists for this message index
                context_key = f"context_{active_id}_{i}"
                if role == "assistant" and context_key in st.session_state:
                    with st.expander("🔍 View Clinical Evidence (Retrieved Context)"):
                        st.info(st.session_state[context_key])

# --- Chat Input & RAG Logic ---
# Extracting values as strings to prevent JSON serialization errors in the embedder
prompt_obj = st.chat_input(
    "Describe symptoms or relevant history...", 
    accept_file="multiple", 
    file_type=["png", "jpg", "jpeg"],
    key="main_input_bar"
)

if prompt_obj:
    # 1. Extract string value and handle display
    user_text = prompt_obj.text if hasattr(prompt_obj, 'text') and prompt_obj.text else str(prompt_obj)
    user_display = user_text
    if hasattr(prompt_obj, 'files') and prompt_obj.files:
        user_display += f" (Attached: {', '.join([f.name for f in prompt_obj.files])})"
    
    # 2. Display User Message & Save to History
    with chat_container:
        st.chat_message("user").markdown(user_display)
    history.add_user_message(user_display)
    
    # 3. Build Chat History Context
    context_history = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in history.messages])
    
    # 4. Generate Assistant Response
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing symptoms against medical database..."):
                # Call the RAG chain
                result = rag_chain.diagnose(patient_query=user_text, chat_history=context_history)
                full_response = result["diagnosis"]
                st.markdown(full_response)
                
                # STORE CONTEXT for the rerun: associate it with the upcoming message index
                current_msg_idx = len(history.messages)
                st.session_state[f"context_{active_id}_{current_msg_idx}"] = result["retrieved_context"]
                
                with st.expander("🔍 View Clinical Evidence (Retrieved Context)"):
                    st.info(result["retrieved_context"])

    # 5. Sync and Rerun
    history.add_ai_message(full_response)
    st.rerun()