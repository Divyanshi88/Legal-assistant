# app.py

import streamlit as st
import time
from datetime import datetime
import os
import sys
import traceback

# Load environment variables first (with fallback for deployment)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available (e.g., in some deployment environments)
    # Environment variables should be set directly in the deployment platform
    pass

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="🏛️ Legal Assistant - Women's Rights & Domestic Violence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Safe import with detailed error handling
def safe_import_rag():
    try:
        # Try to import the module
        from query_database import EnhancedRAGPipeline
        
        # Try to import CHAT_MODELS, with fallback
        try:
            from query_database import CHAT_MODELS
        except ImportError:
            # Fallback CHAT_MODELS if not available
            CHAT_MODELS = {
                "mistral": "mistralai/mistral-7b-instruct",
                "llama": "meta-llama/llama-3-8b-instruct",
                "gpt": "openai/gpt-3.5-turbo"
            }
        
        return EnhancedRAGPipeline, CHAT_MODELS, None
    except Exception as e:
        error_details = f"Error importing RAG pipeline: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, None, error_details

# Try to import RAG pipeline
EnhancedRAGPipeline, CHAT_MODELS, import_error = safe_import_rag()

# If import failed, show error and provide debug info
if import_error:
    st.error("❌ Failed to import RAG pipeline")
    
    with st.expander("🔍 Debug Information"):
        st.text(import_error)
        
        st.write("**Files in current directory:**")
        try:
            files = os.listdir(".")
            for file in files:
                st.write(f"- {file}")
        except Exception as e:
            st.write(f"Could not list files: {e}")
        
        st.write("**Environment variables:**")
        st.write(f"- OPENROUTER_API_KEY: {'✅ Set' if os.getenv('OPENROUTER_API_KEY') else '❌ Not set'}")
        
        st.write("**Python path:**")
        for path in sys.path:
            st.write(f"- {path}")
    
    st.info("Please check that all required files are uploaded and environment variables are set in Streamlit secrets.")
    st.stop()

# Initialize CHAT_MODELS if not available
if not CHAT_MODELS:
    CHAT_MODELS = {
        "mistral": "mistralai/mistral-7b-instruct",
        "llama": "meta-llama/llama-3-8b-instruct",
        "gpt": "openai/gpt-3.5-turbo"
    }

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1, .main-header h3, .main-header p {
        color: white !important;
    }
    
    .suggestion-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .suggestion-card:hover {
        background: #e9ecef;
        border-color: #2a5298;
        transform: translateY(-2px);
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .bot-message {
        background: #f8f9fa;
        color: #333 !important;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        border-left: 4px solid #2a5298;
    }
    
    .bot-message strong {
        color: #333 !important;
    }
    
    .legal-notice {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #333 !important;
    }
    
    .legal-notice h4, .legal-notice p {
        color: #333 !important;
    }
    
    .quick-actions {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
        flex-wrap: wrap;
    }
    
    .action-btn {
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background 0.3s;
    }
    
    .action-btn:hover {
        background: #218838;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stat-item {
        text-align: center;
        color: #333 !important;
    }
    
    .stat-item div {
        color: #333 !important;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ Legal Assistant</h1>
    <h3>Protection of Women from Domestic Violence Act, 2005</h3>
    <p>Get instant legal guidance and support • Available 24/7 • Confidential & Secure</p>
</div>
""", unsafe_allow_html=True)

# Function to initialize pipeline safely
@st.cache_resource
def initialize_pipeline():
    try:
        # Get the first available model
        first_model = list(CHAT_MODELS.values())[0]
        pipeline = EnhancedRAGPipeline(first_model)
        return pipeline, None
    except Exception as e:
        error_msg = f"Failed to initialize pipeline: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return None, error_msg

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pipeline_initialized" not in st.session_state:
    st.session_state.pipeline_initialized = False

# Function to handle question processing
def process_question(question):
    try:
        with st.spinner("🤔 Analyzing your question..."):
            result = st.session_state.pipeline.query_with_sources(
                question,
                mode="legal"
            )
        
        # Add bot response to history
        response_content = f"""{result["answer"]}

---
⏱️ **Response time:** {result['processing_time']:.2f}s  
📄 **Sources consulted:** {len(result['sources'])} legal documents"""
        
        st.session_state.chat_history.append({"role": "assistant", "content": response_content})
        return True
    except Exception as e:
        error_response = f"❌ Sorry, I encountered an error processing your question: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_response})
        return False

# Try to initialize pipeline
if not st.session_state.pipeline_initialized:
    with st.spinner("🔧 Initializing AI system..."):
        pipeline, pipeline_error = initialize_pipeline()
        
        if pipeline_error:
            st.error("❌ Failed to initialize AI pipeline")
            with st.expander("🔍 Pipeline Error Details"):
                st.text(pipeline_error)
            st.stop()
        else:
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_initialized = True
            st.success("✅ AI system initialized successfully!")

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    model_key = st.selectbox("AI Model", list(CHAT_MODELS.keys()), index=0)
    
    # Model change button
    if st.button("🔄 Change Model"):
        try:
            with st.spinner("🔧 Switching model..."):
                new_pipeline = EnhancedRAGPipeline(CHAT_MODELS[model_key])
                st.session_state.pipeline = new_pipeline
                st.success(f"✅ Switched to {model_key}")
        except Exception as e:
            st.error(f"❌ Failed to switch model: {e}")
    
    if st.button("🔄 Reset Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Show current model info
    if hasattr(st.session_state, 'pipeline') and hasattr(st.session_state.pipeline, 'model_name'):
        st.info(f"Current model: {st.session_state.pipeline.model_name}")
    
    # Emergency contacts
    st.markdown("---")
    st.markdown("### 📞 Emergency Contacts")
    st.markdown("""
    **National Commission for Women**  
    📞 7827170170
    
    **Police Helpline**  
    📞 1091 / 1291
    
    **Delhi Women's Cell**  
    📞 (011) 23317004
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Quick Start Section
    if not st.session_state.chat_history:
        st.markdown("### 🚀 Quick Start - How can I help you today?")
        
        # Suggested questions
        suggested_questions = [
            "What is considered domestic violence under the law?",
            "How do I file a complaint for domestic violence?",
            "What is a Protection Order and how do I get one?",
            "What financial help can I get as a victim?",
            "Can I get custody of my children?",
            "Do I have the right to stay in my house?",
            "What happens after I file a case?",
            "Who can help me file a complaint?",
            "What are the different types of orders a Magistrate can pass?",
            "How long does it take to resolve a case?",
            "Can someone else file a complaint on my behalf?",
            "What is the role of a Protection Officer?"
        ]
        
        # Display suggestions in a grid
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            with cols[i % 2]:
                if st.button(f"💬 {question}", key=f"suggest_{i}", use_container_width=True):
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    # Process the question
                    process_question(question)
                    st.rerun()
    
    # Chat Interface
    if st.session_state.chat_history:
        st.markdown("### 💬 Chat History")
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="text-align: right;">
                        <div class="user-message">
                            <strong>You:</strong> {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Clean the message content and preserve line breaks
                    clean_content = message["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>🤖 Legal Assistant:</strong><br>
                        {clean_content}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    # Create a form for better input handling
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "💭 Ask your legal question here...",
            placeholder="Type your question about domestic violence law...",
            key="user_input_form"
        )
        
        send_button = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
        
        # Process user input
        if send_button and user_input.strip():
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process the question
            process_question(user_input)
            st.rerun()

with col2:
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🆘 Emergency Help", use_container_width=True):
        st.info("Call 100 for immediate police assistance")
    if st.button("📋 File Complaint", use_container_width=True):
        st.info("Contact your local police station or magistrate")
    if st.button("👥 Find Support", use_container_width=True):
        st.info("Reach out to NGOs and women's support organizations")
    if st.button("📞 Contact Lawyer", use_container_width=True):
        st.info("Free legal aid is available - call 7827170170")
    
    # Statistics
    st.markdown("### 📊 Quick Stats")
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">3</div>
            <div>Days to start proceedings</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">60</div>
            <div>Days to resolve case</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">24/7</div>
            <div>Help available</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Legal Notice
    st.markdown("### ⚠️ Important Legal Notice")
    st.warning("""
    **Important Legal Notice**
    
    This chatbot provides information based on the Protection of Women from Domestic Violence Act, 2005. For specific legal advice, please consult with a qualified lawyer.
    
    **In case of emergency, contact police immediately at 100.**
    """)
    
    # Key Features
    st.markdown("### ✨ Key Features")
    st.markdown("""
    - 🔒 **Confidential**: Your questions are private and secure
    - ⚡ **Instant**: Get immediate legal guidance
    - 📚 **Accurate**: Based on official legal documents
    - 🆓 **Free**: No charges for legal information
    - 🌐 **24/7**: Available anytime, anywhere
    - 📱 **Mobile-friendly**: Works on all devices
    """)
    
    # Recent Topics
    if st.session_state.chat_history:
        st.markdown("### 📈 Your Recent Topics")
        recent_topics = []
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                topic = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                recent_topics.append(f"• {topic}")
        
        for topic in recent_topics:
            st.markdown(topic)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem; color: #333;">
    <h4 style="color: #333;">🛡️ Your Rights Matter</h4>
    <p style="color: #333;">This service is provided to help you understand your legal rights under the Protection of Women from Domestic Violence Act, 2005.</p>
    <p style="color: #333;"><strong>Remember:</strong> You are not alone. Help is available.</p>
    <p style="font-size: 0.9rem; color: #666;">
        🔐 All conversations are confidential • 📱 Available on mobile • 🆓 Completely free service
    </p>
</div>
""", unsafe_allow_html=True)