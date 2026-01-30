import streamlit as st
import ollama

st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("✈️ Travel Itinerary Assistant")

# --- 1. Sidebar Configuration ---
with st.sidebar:
    st.header("Travel Agent Settings")
    
    try:
        models_info = ollama.list()
        model_names = [m['model'] for m in models_info['models']]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        model_names = []
    
    selected_model = st.selectbox("Select a Model", model_names, index=0 if model_names else None)

    # --- HARDCODED SYSTEM PROMPT ---
    # We put your travel instructions directly in the 'value' attribute
    travel_persona = (
        "You are an expert Travel Itinerary Planner AI. Your sole purpose is to create detailed, "
        "efficient, and comfortable travel plans for users.\n\n"
        "**YOUR PROTOCOL:**\n"
        "1. Data Gathering: If the user has not provided them yet, you must ask for: "
        "The Destination, the Start Date, and the End Date.\n"
        "2. Itinerary Generation: Once dates/location are known, provide a day-by-day plan with "
        "Morning, Afternoon, and Evening activities.\n"
        "3. Transport: Suggest the most efficient mode of transport between spots, balancing "
        "cost and comfort (e.g., Metro vs. Uber).\n"
        "4. Scope: You ONLY answer travel-related queries. If asked about anything else, say: "
        "'I specialize only in travel itineraries. Let's get back to planning your trip!'"
    )

    # Displaying it in the sidebar so you can see it's working
    system_prompt = st.text_area(
        "System Instructions",
        value=travel_persona,  # <--- The hardcoded value is here
        height=300
    )
    
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 2. Chat Logic (Same as before) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tell me where you want to go!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Merging the prompt with the history for the API call
            messages_payload = [
                {"role": "system", "content": system_prompt}
            ] + st.session_state.messages
            
            stream = ollama.chat(
                model=selected_model,
                messages=messages_payload,
                stream=True,
            )
            
            for chunk in stream:
                content = chunk['message']['content']
                full_response += content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})