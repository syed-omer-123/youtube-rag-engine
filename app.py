import os
import streamlit as st
from dotenv import load_dotenv
from main import run_production_pipeline, answer_dynamic_query

# Load local .env variables at the absolute start of runtime execution
load_dotenv()

# Configure layout view
st.set_page_config(page_title="YouTube RAG Engine", page_icon="🤖", layout="centered")

st.title("🤖 Production YouTube RAG Engine")
st.write("Explore our pre-cached video for free, or connect your own API key to run a live pipeline on any new video link.")

# --- SIDEBAR ACCESS SELECTION ---
st.sidebar.title("⚙️ Engine Configurations")
demo_mode = st.sidebar.radio(
    "Select Mode:",
    ["1. Free Interactive Demo Mode", "2. Live Custom Video Mode"]
)

st.sidebar.markdown("---")

# Setup access rules based on selected execution layer
if "Free Interactive Demo Mode" in demo_mode:
    st.sidebar.info("💡 Running via pre-cached FAISS database. No personal API Key required.")
    # Read the master cloud secret key automatically hidden inside Streamlit's environment
    active_api_key = os.environ.get("OPENAI_API_KEY")
    video_url = "https://www.youtube.com/watch?v=JCOb1w_LTOg"
    st.text_input("Target Video URL", video_url, disabled=True)
else:
    st.sidebar.subheader("🔐 API Key Authorization")
    active_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password", placeholder="sk-...")
    video_url = st.text_input("Enter Any YouTube Watch URL", placeholder="https://www.youtube.com/watch?v=...")

# --- PIPELINE ACTIONS ---
if active_api_key:
    # Set key context into operational system memory
    os.environ["OPENAI_API_KEY"] = active_api_key


# --- HELPER FUNCTION FOR BULLETPROOF ERROR HANDLING ---
def handle_rag_errors(error_exception, mode):
    """
    Interceptors backend API errors to elegantly guide the user 
    instead of printing raw code exceptions.
    """
    error_str = str(error_exception).lower()
    
    # CASE 1: The personal showcase budget limit is reached
    if "insufficient_quota" in error_str or "quota" in error_str or "limit exceeded" in error_str:
        if "Free Interactive" in mode:
            st.warning(
                "⚠️ **Demo Budget Cap Reached:** The monthly showcase budget allocated for this public portfolio "
                "piece has been fully exhausted by visitors. \n\n"
                "**How to continue testing:** Simply switch the **Select Mode** in the sidebar to "
                "**'2. Live Custom Video Mode'** and input your personal OpenAI API Key to run unlimited questions!"
            )
        else:
            st.error("🛑 The OpenAI API Key you provided has exceeded its billing quota or usage limits. Please check your account status.")
            
    # CASE 2: The user entered a broken, fake, or mistyped API Key
    elif "invalid_api_key" in error_str or "401" in error_str or "incorrect api key" in error_str:
        st.error(
            "🔐 **Invalid API Key Detected:** The authentication token provided could not be verified by OpenAI. "
            "Please check for typos or ensure you are passing a valid, active secret key string starting with 'sk-'."
        )
        
    # CASE 3: Any other unexpected system failure (Network drop, YouTube API block, etc.)
    else:
        st.error(f"⚙️ System Pipeline Error: {error_exception}")


st.markdown("### 📋 Core Executive Summary")
if st.button("Generate Structured Executive Report", type="primary"):
    if not active_api_key:
        st.error("🛑 Please provide an OpenAI API Key in the sidebar to execute pipeline structures.")
    elif "Live Custom" in demo_mode and ("watch?v=" not in video_url):
        st.error("🛑 Please enter a valid YouTube link.")
    else:
        with st.spinner("Analyzing text chunk context distributions..."):
            try:
                final_output = run_production_pipeline(video_url)
                st.success("Analysis Complete!")
                st.subheader(f"🎬 Title: {final_output.video_title}")
                st.write(f"**Macro Thesis:** {final_output.main_takeaway}")
                
                st.write("#### 📌 Core Pillars")
                for pillar in final_output.key_pillars:
                    st.write(f"- {pillar}")
                
                st.write("#### 🚀 Operational Action Steps")
                for step in final_output.actionable_steps:
                    st.write(f"- {step}")
                
                st.write("#### 💾 Verifiable JSON Output Schema")
                st.json(final_output.model_dump_json())
            except Exception as e:
                handle_rag_errors(e, demo_mode)

st.markdown("---")

# --- CONVERSATIONAL CHAT INTERACTION LAYER ---
st.markdown("### 💬 Ask Any Question About This Video")
user_chat_query = st.text_input("Type your question here:", placeholder="e.g., What does the speaker say about vulnerability or delusion?")

if st.button("Submit Question"):
    if not active_api_key:
        st.error("🛑 Please ensure an API Key is loaded or available to generate answer text strings.")
    elif not user_chat_query:
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching localized FAISS index frames and evaluating answers..."):
            try:
                answer_result = answer_dynamic_query(user_chat_query)
                st.markdown("#### 🤖 Response Summary:")
                st.info(answer_result)
            except Exception as e:
                handle_rag_errors(e, demo_mode)