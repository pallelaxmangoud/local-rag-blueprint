import streamlit as st
import os
import fitz  # PyMuPDF
from groq import Groq

st.set_page_config(page_title="Cloud AI Document Assistant", layout="wide")

st.title("📄 Cloud-Powered AI Document Assistant")
st.write("Interact with your PDF documents seamlessly using ultra-fast cloud inference.")

# Secure input for API Key in the sidebar
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key:", type="password")

# Sidebar for document uploads
with st.sidebar:
    st.header("📂 Document Center")
    uploaded_file = st.file_uploader("Upload any PDF document:", type=["pdf"])

if uploaded_file:
    # Save file locally
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

    # Extract text content safely using PyMuPDF
    @st.cache_data
    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        sections = []
        current_text = ""
        
        for page in doc:
            current_text += page.get_text()
            if len(current_text) > 1500:
                sections.append(current_text.strip())
                current_text = ""
        if current_text:
            sections.append(current_text.strip())
        return sections

    sections = extract_text_from_pdf("temp_doc.pdf")
    st.success(f"Context layer built! Loaded {len(sections)} sections.")

    # User Query Section
    user_query = st.text_input("💬 Ask a question about your document:", placeholder="Summarize the pdf...")
    
    if st.button("Get Answer", type="primary") and user_query:
        if not api_key:
            st.error("Please provide your Groq API Key in the sidebar to run cloud queries.")
        else:
            best_context = sections[0] if sections else ""
            for sec in sections:
                keywords = [word for word in user_query.lower().split() if len(word) > 3]
                if any(k in sec.lower() for k in keywords):
                    best_context = sec
                    break

            st.info("⚡ Cloud AI is processing your request...")
            
            try:
                client = Groq(api_key=api_key)
                
                # Updated to active, supported model
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": f"Use this context to answer: {best_context}"},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                
                answer = completion.choices[0].message.content
                st.subheader("🤖 Response")
                st.write(answer)
                
            except Exception as e:
                st.error(f"An API error occurred: {e}")
else:
    st.info("Please upload a PDF document in the sidebar to begin.")