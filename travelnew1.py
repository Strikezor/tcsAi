import streamlit as st
import ollama
import time
from datetime import date

st.set_page_config(page_title="AI Travel Planner", layout="wide", page_icon="✈️")

# --- 1. SESSION STATE ---
if "itinerary_ready" not in st.session_state:
    st.session_state.itinerary_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR (Model Selection) ---
with st.sidebar:
    st.header("Settings")
    try:
        models_info = ollama.list()
        model_names = [m['model'] for m in models_info['models']]
    except:
        model_names = ["llama-3.2-3b-it:latest"]
    
    selected_model = st.selectbox("LLM Engine", model_names)
    if st.button("New Trip / Reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- 3. INPUT PHASE ---
if not st.session_state.itinerary_ready:
    st.title("✈️ AI Travel Itinerary Planner")
    with st.form("travel_details_form"):
        origin = st.text_input("Your Current Location", placeholder="City, Country")
        col1, col2 = st.columns(2)
        with col1:
            dest = st.text_input("Destination", placeholder="e.g. Rome, Italy")
        with col2:
            dates = st.date_input("Travel Window", value=(date.today(), date.today()))
            
        if st.form_submit_button("Generate My Itinerary"):
            if not origin or not dest or len(dates) < 2:
                st.warning("Please fill in all fields.")
            else:
                st.session_state.update({"origin": origin, "dest": dest, "start": dates[0], "end": dates[1], "itinerary_ready": True})
                # Hidden first prompt
                st.session_state.messages.append({"role": "user", "content": f"Traveling from {origin} to {dest} ({dates[0]} to {dates[1]})."})
                st.rerun()

# --- 4. CHAT PHASE ---
else:
    st.title(f"📍 {st.session_state.origin} → {st.session_state.dest}")
    st.info(f"📅 Dates: {st.session_state.start} to {st.session_state.end}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            # Use st.empty to show loading/timer text BEFORE the response
            status_text = st.empty()
            response_placeholder = st.empty()
            
            start_time = time.time()
            full_response = ""
            
            # Simple "Loading" display
            status_text.markdown("⏳ *Analyzing routes and attractions...*")
            
            try:
                stream = ollama.chat(
                    model=selected_model,
                    messages=[{"role": "system", "content": "You are a Travel Planner. Use the user's current location to suggest transport. Plan daily activities. Be concise. Only travel topics."}] + st.session_state.messages,
                    stream=True,
                )
                
                for chunk in stream:
                    if not full_response:
                        status_text.empty()
                    
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")
                
                elapsed = round(time.time() - start_time, 2)
                response_placeholder.markdown(full_response)
                st.caption(f"⚡ Generated in {elapsed}s")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {e}")

    if prompt := st.chat_input("Ask for adjustments..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()