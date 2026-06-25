import streamlit as st
import os
import fitz  # PyMuPDF
from llama_cpp import Llama
import threading

# Initialize a global thread lock to prevent streamlit reruns from crashing the C-subprocesses
if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()

st.set_page_config(page_title="Local Document AI", layout="wide")

st.title("📄 Local AI Document Assistant")
st.write("Interact with your PDF documents completely offline and private using local LLM execution.")

# Initialize the LLM with strict low-memory constraints
@st.cache_resource
def load_llm():
    model_path = os.path.join("models", "qwen2.5-7b-instruct-q4_k_m.gguf")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}. Please check your models folder.")
        return None
    # Lowered n_ctx to 2048 to significantly decrease RAM consumption and prevent crashes
    return Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)

llm = load_llm()

# Sidebar for document uploads
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload any PDF document:", type=["pdf"])

if uploaded_file:
    # Save file locally
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    # Extract text content
    @st.cache_data
    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        sections = []
        current_text = ""
        
        for page in doc:
            current_text += page.get_text()
            if len(current_text) > 1000:
                sections.append(current_text.strip())
                current_text = ""
        if current_text:
            sections.append(current_text.strip())
        return sections

    sections = extract_text_from_pdf("temp_doc.pdf")
    st.success(f"Context layer built! Loaded {len(sections)} distinct document sections.")

    # User Query Section
    user_query = st.text_input("💬 Ask a question about your document:", placeholder="Summarise the pdf...")
    
    if st.button("Get Answer", type="primary") and user_query:
        if llm is None:
            st.error("LLM is not loaded.")
        else:
            # Simple keyword matching to pull the most relevant context chunk
            best_context = sections[0] if sections else ""
            for sec in sections:
                keywords = [word for word in user_query.lower().split() if len(word) > 3]
                if any(k in sec.lower() for k in keywords):
                    best_context = sec
                    break

            st.info("🧠 AI is evaluating document context...")
            
            # Format custom instruction prompt safely
            prompt = f"<|im_start|>system\nYou are a helpful assistant. Use this context to answer: {best_context}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
            
            # Thread-locked generation to guarantee Streamlit stability
            with st.session_state.lock:
                try:
                    response = llm(
                        prompt,
                        max_tokens=512,
                        temperature=0.7,
                        stop=["<|im_end|>"],
                        echo=False
                    )
                    answer = response["choices"][0]["text"].strip()
                    
                    st.subheader("🤖 Response")
                    st.write(answer)
                except Exception as e:
                    st.error(f"An unexpected evaluation error occurred: {e}")
else:
    st.info("Please upload a PDF document in the sidebar to begin.")