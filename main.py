import os
import re
import tempfile
import threading
import pymupdf4llm
import streamlit as st
from llama_cpp import Llama

# Prevent multi-threaded collisions when calling the underlying C-library
MODEL_LOCK = threading.Lock()

st.set_page_config(
    page_title="Mozilla-Inspired Local AI",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling to match a premium Mozilla-inspired look
st.markdown("""
    <style>
    .main {
        background-color: #0f111a;
        color: #e6e6e6;
    }
    .stTextInput>div>div>input {
        background-color: #1a1c24;
        color: #ffffff;
        border: 1px solid #3f445b;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #ff4a5a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff6b7b;
        box-shadow: 0px 0px 10px rgba(255, 74, 90, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Mozilla-Inspired Local Document Q&A")
st.write("Interact with your PDF documents completely offline and private using local LLM execution.")

MODEL_PATH = "models/qwen2.5-7b-instruct-q4_k_m.gguf"

@st.cache_resource
def load_model():
    """Loads the model with a highly optimized memory footprint (n_ctx=2048) to prevent crashes."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at `{MODEL_PATH}`! Please check your models folder.")
        st.stop()
    try:
        # Reduced n_ctx from 4096 to 2048 to save ~4GB of RAM and prevent Out Of Memory crashes
        return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

@st.cache_data
def process_pdf(pdf_path):
    """Parses PDF pages, converts to clean Markdown, and segments by headers."""
    try:
        markdown_content = pymupdf4llm.to_markdown(pdf_path)
        # Use headers (# Header Name) to segment the document dynamically
        header_pattern = r'(^#{1,6}\s+.*$)'
        tokens = re.split(header_pattern, markdown_content, flags=re.MULTILINE)
        
        document_sections = {}
        current_section_title = "Document Overview"
        document_sections[current_section_title] = ""
        
        for token in tokens:
            if re.match(header_pattern, token):
                current_section_title = token.strip()
                document_sections[current_section_title] = ""
            else:
                document_sections[current_section_title] += token
        
        # Filter out empty sections
        document_sections = {k: v for k, v in document_sections.items() if v.strip()}
        
        # If no markdown headers were found, treat the whole document as one segment
        if not document_sections:
            document_sections["Document Content"] = markdown_content
            
        return document_sections
    except Exception as e:
        st.error(f"Failed to process PDF file: {str(e)}")
        return None

with st.spinner("🧠 Booting up your local AI brain (may take 20-30 seconds on cold start)..."):
    llm = load_model()

st.sidebar.markdown("### 📁 Document Center")
uploaded_file = st.sidebar.file_uploader("Upload any PDF document:", type=["pdf"])

pdf_to_use = None
temp_file_path = None

# Logic to handle uploaded file OR fall back to pre-saved default PDF
if uploaded_file is not None:
    # Save the uploaded file in a temporary folder to let MuPDF parse it
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_to_use = temp_file_path
    st.sidebar.success(f"Loaded: `{uploaded_file.name}`")
else:
    # Fallback default file
    default_pdf = "sample_document.pdf"
    if os.path.exists(default_pdf):
        pdf_to_use = default_pdf
        st.sidebar.info(f"Using default document: `{default_pdf}`")
    else:
        st.sidebar.warning("⚠️ No document uploaded yet. Drag and drop any PDF file above to begin!")

if pdf_to_use:
    sections = process_pdf(pdf_to_use)
    
    if sections:
        headers_list = list(sections.keys())
        st.success(f"🤖 Context layer built! Loaded **{len(headers_list)}** distinct document sections.")
        
        # Clean up temporary file safely after loading context
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        
        user_query = st.text_input("💬 Ask a question about your document:")
        submit_clicked = st.button("Get Answer")
        
        if user_query or submit_clicked:
            if not user_query.strip():
                st.warning("Please type a question before submitting!")
            else:
                # Add a spinner to give the user visual feedback that it is working
                with st.spinner("AI is evaluating document context..."):
                    try:
                        # Establish a model lock to protect inference calculations from threading crashes
                        with MODEL_LOCK:
                            formatted_headers = "\n".join([f"- {header}" for header in headers_list])
                            sys_instruct_1 = (
                                "Match the user query to the single most relevant section header. "
                                "Reply ONLY with the exact text of the matched header."
                            )
                            prompt_1 = (
                                f"<|im_start|>system\n{sys_instruct_1}<|im_end|>\n"
                                f"<|im_start|>user\nAvailable Sections:\n{formatted_headers}\n\n"
                                f"Query: {user_query}\n\n"
                                f"Target Header:<|im_end|>\n<|im_start|>assistant\n"
                            )
                            
                            out_1 = llm(prompt_1, max_tokens=120, temperature=0.0)
                            predicted_header = out_1['choices'][0]['text'].strip()
                            
                            # Match prediction against our keys safely
                            matched_section = headers_list[0]
                            for header in headers_list:
                                if header.lower() in predicted_header.lower() or predicted_header.lower() in header.lower():
                                    matched_section = header
                                    break
                            
                            st.info(f"🎯 **Target Section Selected:** `{matched_section}`")
                            
                            context_text = sections[matched_section]
                            sys_instruct_2 = (
                                f"Answer the user question strictly using the provided context from "
                                f"the section '{matched_section}'. Keep your answer detailed, clear, and professional."
                            )
                            prompt_2 = (
                                f"<|im_start|>system\n{sys_instruct_2}<|im_end|>\n"
                                f"<|im_start|>user\nContext:\n{context_text}\n\n"
                                f"Question: {user_query}\n\n"
                                f"Answer:<|im_end|>\n<|im_start|>assistant\n"
                            )
                            
                            out_2 = llm(prompt_2, max_tokens=512, temperature=0.1)
                            final_response = out_2['choices'][0]['text'].strip()
                            
                        st.markdown("---")
                        st.markdown("### 🤖 Response")
                        st.write(final_response)
                        
                    except Exception as inference_error:
                        st.error(f"⚠️ Calculation error: {str(inference_error)}")
                        st.info("💡 Try restarting Streamlit. Memory resources might be heavily restricted right now.")
    else:
        st.error("Could not extract sections from the PDF. Make sure it contains readable text layers.")
else:
    st.info("💡 **Getting started:** Drag and drop any PDF file in the sidebar to begin your local document Q&A.")