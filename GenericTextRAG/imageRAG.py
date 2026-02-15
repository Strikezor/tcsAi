import streamlit as st
import base64
import httpx
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()



# --- Configuration & Setup ---
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Client setup to bypass SSL verification (if required by your environment)
client = httpx.Client(verify=False)

# Page Config
st.set_page_config(page_title="Vision Analyzer", layout="centered")
st.title("👁️ Vision-Based Chat")

# --- Helper Functions ---

def encode_image(uploaded_file):
    """Convert uploaded image to base64 string."""
    image = Image.open(uploaded_file)
    buffered = BytesIO()
    # Convert to RGB if necessary (handles RGBA/transparent images)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_vision_response(base64_image, user_query):
    """Sends the image and query to the Vision Model."""
    # Using Llama 3.2 90B Vision from your provided model list
    llm = ChatOpenAI(
        base_url="https://genailab.tcs.in",
        model="azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=client,
        max_tokens=1024
    )

    # Creating a multimodal message
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": f"Analyze this image and answer the following based only on the basis of the image provided: {user_query}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )
    
    response = llm.invoke([message])
    return response.content

# --- Main UI Logic ---

if "vision_messages" not in st.session_state:
    st.session_state.vision_messages = []

if "image_b64" not in st.session_state:
    st.session_state.image_b64 = None

# Sidebar for Image Upload
with st.sidebar:
    st.header("📸 Image Upload")
    uploaded_img = st.file_uploader("Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_img:
        # Display a small preview in sidebar
        st.image(uploaded_img, caption="Preview", use_container_width=True)
        # Encode image to base64 for processing
        st.session_state.image_b64 = encode_image(uploaded_img)
        st.success("Image uploaded successfully!")

# Chat Interface
st.divider()

for msg in st.session_state.vision_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the image..."):
    if not st.session_state.image_b64:
        st.error("Please upload an image first!")
    else:
        # 1. Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.vision_messages.append({"role": "user", "content": prompt})

        # 2. Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing image..."):
                try:
                    answer = get_vision_response(st.session_state.image_b64, prompt)
                    st.markdown(answer)
                    st.session_state.vision_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Vision error: {e}")