import streamlit as st
import ollama

# 1. Page Configuration
st.set_page_config(page_title="Local Ollama Chat", layout="wide")
st.title("💬 Local Ollama Chatbot")

# 2. Sidebar for Model Selection
# We fetch the models dynamically from your local Ollama instance
try:
    models_info = ollama.list()
    # ollama.list() returns a dictionary with a 'models' key
    # We extract the 'model' name from each item
    model_names = [m['model'] for m in models_info['models']]
except Exception as e:
    st.error(f"Failed to fetch models: {e}")
    model_names = []

with st.sidebar:
    st.header("Settings")
    # Default to the first model if available, otherwise empty
    selected_model = st.selectbox("Select a Model", model_names, index=0 if model_names else None)
    
    system_prompt = st.text_area("System Instructions",
                                 value = "",
                                height = 100)
    
    st.divider()
    
    # Optional: Button to clear chat history
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6. Generate Assistant Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Call Ollama with the selected model and full history
        try:
            messages_payload = [
                {"role":"system","content":system_prompt}
            ] + st.session_state.messages
            
            stream = ollama.chat(
                model=selected_model,
                messages=st.session_state.messages,
                stream=True,
            )
            
            # Stream the response chunk by chunk
            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Error generating response."

    # Add assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})