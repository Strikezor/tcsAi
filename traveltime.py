import streamlit as st
import ollama
import time

st.set_page_config(page_title="AI Travel Planner", layout="wide", page_icon="✈️")
st.title("✈️ Travel Itinerary Assistant")

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("Travel Agent Settings")
    
    try:
        models_info = ollama.list()
        model_names = [m['model'] for m in models_info['models']]
    except Exception as e:
        st.error("Ollama not detected. Please run 'ollama serve'")
        model_names = []
    
    selected_model = st.selectbox("Select a Model", model_names, index=0 if model_names else None)

    travel_persona = (
        "You are an expert Travel Itinerary Planner AI. Your sole purpose is to create detailed, "
        "efficient, and comfortable travel plans for users.\n\n"
        "**YOUR PROTOCOL:**\n"
        "1. Data Gathering: Ask for Destination, Start Date, and End Date if missing.\n"
        "2. Itinerary: Provide a day-by-day plan (Morning, Afternoon, Evening).\n"
        "3. Transport: Suggest efficient modes balancing cost and comfort.\n"
        "4. Scope: ONLY travel queries. Refuse anything else politely."
    )

    system_prompt = st.text_area("System Instructions", value=travel_persona, height=250)
    
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 2. Initialize Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. Chat Logic with Timer & Progress ---
if prompt := st.chat_input("Tell me where you want to go!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Start the clock
        start_time = time.time()
        
        with st.status("Travel Assistant is thinking...", expanded=True) as status:
            st.write(f"Connecting to {selected_model}...")
            
            response_placeholder = st.empty()
            full_response = ""

            try:
                messages_payload = [
                    {"role": "system", "content": system_prompt}
                ] + st.session_state.messages
                
                stream = ollama.chat(
                    model=selected_model,
                    messages=messages_payload,
                    stream=True,
                )
                
                # We stop the status box the moment the first word arrives
                first_chunk = True
                for chunk in stream:
                    if first_chunk:
                        # Calculate time to first token
                        end_time = time.time()
                        elapsed_time = round(end_time - start_time, 2)
                        
                        status.update(
                            label=f"Itinerary Generated in {elapsed_time}s", 
                            state="complete", 
                            expanded=False
                        )
                        first_chunk = False
                    
                    content = chunk['message']['content']
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
            
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})