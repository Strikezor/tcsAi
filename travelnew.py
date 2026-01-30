import streamlit as st
import ollama
import time
from datetime import date

st.set_page_config(page_title="AI Travel Planner", layout="wide", page_icon="✈️")

# --- 1. SESSION STATE INITIALIZATION ---
if "itinerary_ready" not in st.session_state:
    st.session_state.itinerary_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR ---
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

# --- 3. INPUT PHASE (Displayed only if itinerary not yet started) ---
if not st.session_state.itinerary_ready:
    st.title("✈️ AI Travel Itinerary Planner")
    st.markdown("Please provide your trip details below to get started.")
    
    with st.form("travel_details_form"):
        # Current Location Field Added
        origin = st.text_input("Your Current Location", placeholder="City, Country")
        
        col1, col2 = st.columns(2)
        with col1:
            dest = st.text_input("Destination", placeholder="e.g. Rome, Italy")
        with col2:
            dates = st.date_input("Travel Window", value=(date.today(), date.today()))
            
        submit = st.form_submit_button("Generate My Itinerary")
        
        if submit:
            if not origin or not dest or len(dates) < 2:
                st.warning("Please fill in your location, destination, and select a date range.")
            else:
                # Save data and toggle phase
                st.session_state.itinerary_ready = True
                st.session_state.origin = origin
                st.session_state.dest = dest
                st.session_state.start_date = dates[0]
                st.session_state.end_date = dates[1]
                
                # Create the first hidden prompt to kickstart the AI
                initial_query = (
                    f"I am traveling from {origin} to {dest} from {dates[0]} to {dates[1]}. "
                    "Please provide a detailed itinerary and suggest the best transport."
                )
                st.session_state.messages.append({"role": "user", "content": initial_query})
                st.rerun()

# --- 4. CHAT PHASE (Visible only after form submission) ---
else:
    st.title(f"📍 {st.session_state.origin} → {st.session_state.dest}")
    st.info(f"📅 Dates: {st.session_state.start_date} to {st.session_state.end_date}")

    # Display Chat History
    for message in st.session_state.messages:
        # Hide the very first automated prompt from the UI for a cleaner look
        if message["content"].startswith(f"I am traveling from {st.session_state.origin}"):
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Auto-trigger response for the first prompt
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            system_msg = {
                "role": "system", 
                "content": (
                    "You are a Travel Planner. Use the user's current location to suggest travel "
                    "modes (flights, trains, etc.) to the destination. Provide a day-by-day "
                    "breakdown. Balance cost and comfort. Only answer travel-related queries."
                )
            }
            
            t_start = time.time()
            try:
                stream = ollama.chat(
                    model=selected_model,
                    messages=[system_msg] + st.session_state.messages,
                    stream=True,
                )
                
                for chunk in stream:
                    full_response += chunk['message']['content']
                    response_placeholder.markdown(full_response + "▌")
                
                t_end = time.time()
                response_placeholder.markdown(full_response)
                st.caption(f"⏱️ Response generated in {round(t_end - t_start, 2)}s")
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Connection Error: {e}")

    # Subsequent chat interaction
    if prompt := st.chat_input("Ask for adjustments or more details..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()