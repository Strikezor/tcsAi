import streamlit as st
import tempfile
import os
import httpx
import base64
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydub import AudioSegment

# External libraries for document parsing
import PyPDF2
from docx import Document

# Load env
load_dotenv()

# Setup HTTP client to bypass SSL (Use with caution)
http_client = httpx.Client(verify=False)

# 1. LangChain Chat Client (Model from your list)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="genailab-maas-gpt-4o", 
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

# 2. Standard OpenAI Client (For Whisper)
audio_client = OpenAI(
    base_url="https://genailab.tcs.in",
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

st.set_page_config(page_title="Ultimate Multimodal Chat", layout="centered")
st.title("📄 Multi-Format AI Assistant")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_context" not in st.session_state:
    st.session_state.file_context = ""
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "context_ready" not in st.session_state:
    st.session_state.context_ready = False

# --- Helper Functions ---

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def transcribe_large_audio(file_path):
    """
    Splits large audio files into 10-minute chunks to bypass the 25MB limit.
    """
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    # If smaller than 24MB, send directly
    if file_size_mb <= 24:
        with open(file_path, "rb") as f:
            t = audio_client.audio.transcriptions.create(
                model="azure/genailab-maas-whisper",
                file=f
            )
        return t.text

    st.info(f"Large file detected ({file_size_mb:.2f} MB) — splitting into 10-minute segments...")
    
    # Load audio using pydub
    audio = AudioSegment.from_file(file_path)
    ten_minutes = 10 * 60 * 1000  # pydub works in milliseconds
    transcripts = []

    # Calculate total chunks for progress bar
    total_chunks = len(audio) // ten_minutes + (1 if len(audio) % ten_minutes > 0 else 0)
    progress_bar = st.progress(0)

    for i, chunk in enumerate(audio[::ten_minutes]):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_chunk:
            # Export chunk as a valid audio file
            chunk.export(tmp_chunk.name, format="mp3")
            
            with open(tmp_chunk.name, "rb") as f:
                t = audio_client.audio.transcriptions.create(
                    model="azure/genailab-maas-whisper",
                    file=f
                )
                transcripts.append(t.text)
            
            os.remove(tmp_chunk.name)
            
            # Update progress
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)
            
    progress_bar.empty()
    return " ".join(transcripts)

# --- Sidebar: File Upload & Context Injection ---
with st.sidebar:
    st.header("Files & Media")
    uploaded_file = st.file_uploader(
        "Upload Image, Audio, PDF, or DOCX", 
        type=["txt", "png", "jpg", "jpeg", "mp3", "wav", "m4a", "pdf", "docx"]
    )

    # Logic to process file immediately upon upload
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        # 1. Handle Images
        if file_ext in ["png", "jpg", "jpeg"]:
            # Only process if not already processed
            if not st.session_state.context_ready:
                st.image(uploaded_file)
                base64_image = base64.b64encode(uploaded_file.read()).decode("utf-8")
                st.session_state.image_data = f"data:image/{file_ext};base64,{base64_image}"
                st.session_state.context_ready = True
                st.success("Image added to context! You can now ask questions about it.")

        # 2. Handle Audio
        elif file_ext in ["mp3", "wav", "m4a"]:
            if not st.session_state.context_ready:
                if st.button("Start Transcription"):
                    with st.spinner("Transcribing audio (this may take a moment)..."):
                        # Save to temp file for pydub/processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            temp_path = tmp.name
                        
                        try:
                            st.session_state.file_context = transcribe_large_audio(temp_path)
                            st.session_state.context_ready = True
                            st.success("Transcription complete! Context added.")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

        # 3. Handle Documents (PDF / DOCX / TXT)
        elif file_ext in ["pdf", "docx", "txt"]:
            if not st.session_state.context_ready:
                with st.spinner(f"Extracting text from {file_ext.upper()}..."):
                    if file_ext == "pdf":
                        st.session_state.file_context = extract_text_from_pdf(uploaded_file)
                    elif file_ext == "docx":
                        st.session_state.file_context = extract_text_from_docx(uploaded_file)
                    else: # txt
                        st.session_state.file_context = uploaded_file.read().decode("utf-8")
                    
                    st.session_state.context_ready = True
                    st.success(f"{file_ext.upper()} content added to context!")

    if st.session_state.context_ready:
        st.info("Context is active. Ask your questions below.")
        if st.button("Clear Context & Reset"):
            st.session_state.file_context = ""
            st.session_state.image_data = None
            st.session_state.context_ready = False
            st.rerun()

# --- Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your files..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare Payload
    # We always inject the context if it exists
    context_text = st.session_state.file_context if st.session_state.file_context else "No document context provided."
    
    full_user_message = f"""
    User Query: {prompt}
    
    [BACKGROUND CONTEXT FROM UPLOADED FILE]:
    {context_text}
    """
    
    content_list = [{"type": "text", "text": full_user_message}]
    
    # Append image if available
    if st.session_state.image_data:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": st.session_state.image_data}
        })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke([
                    SystemMessage(content="You are a helpful AI assistant. Answer the user's question based strictly on the provided document context or image. If the answer is not in the context, say so."),
                    HumanMessage(content=content_list)
                ])
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                st.error(f"An error occurred: {e}")