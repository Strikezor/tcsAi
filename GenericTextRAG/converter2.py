import streamlit as st
import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration & Setup ---
load_dotenv()

st.set_page_config(
    page_title="Legacy Code Converter",
    layout="wide",
    page_icon="🔄"
)

# --- Specific API Client Setup (Strict Adherence) ---
try:
    client_httpx = httpx.Client(verify=False)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://genailab.tcs.in",
        http_client=client_httpx
    )
except Exception as e:
    st.error(f"Failed to initialize API Client: {e}")
    st.stop()

MODEL_ID = "genailab-maas-gpt-4o"

# --- Helper Functions ---

def analyze_and_detect(source_code):
    """
    Step 1: Analyzes the code logic AND detects the language.
    """
    system_prompt = (
        "You are an expert Code Analyst. "
        "Your task is to analyze the provided code. "
        "1. Identify the programming language strictly. "
        "2. Analyze the core logic and business rules. "
        "Output format must be exactly:\n"
        "DETECTED_LANGUAGE: <Language Name>\n"
        "SUMMARY: <Technical Summary>"
    )
    
    user_prompt = f"Code:\n{source_code}"

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during Analysis phase: {e}")
        return None

def generate_target_code(source_code, analysis_summary, target_lang):
    """
    Step 2: Generates the target code using the analysis summary.
    """
    system_prompt = (
        f"You are an expert Polyglot Developer. Convert the provided code to {target_lang}. "
        "Use the provided technical analysis to ensure logical equivalence. "
        "IMPORTANT: Output ONLY the raw code. Do not include markdown formatting (like ```python), "
        "explanations, or conversational text."
    )

    user_prompt = (
        f"Context & Analysis:\n{analysis_summary}\n\n"
        f"Original Source Code:\n{source_code}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during Generation phase: {e}")
        return None

# --- UI Layout ---

st.title("🔄 Legacy Code Converter")
st.markdown("### Intelligent Two-Pass Conversion with Language Detection")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Source Code")
    # Added common legacy languages to the list
    source_lang_selection = st.selectbox(
        "Select Source Language (Optional)",
        ["Auto-Detect", "COBOL", "Fortran", "C++", "Java", "Python", "Pascal", "VB.NET", "Perl"],
        index=0
    )
    
    source_code = st.text_area(
        "Paste Legacy Code Here:",
        height=500,
        placeholder="       IDENTIFICATION DIVISION.\n       PROGRAM-ID. HELLO-WORLD..."
    )

with col2:
    st.subheader("🚀 Target Output")
    target_lang = st.selectbox(
        "Target Language",
        ["Python", "Java", "Go", "Rust", "JavaScript", "C#"],
        index=0
    )
    
    convert_btn = st.button("Convert Code", type="primary", use_container_width=True)
    
    # Placeholders for dynamic content
    warning_area = st.empty()
    output_area = st.empty()

# --- Main Logic ---

if convert_btn:
    if not source_code.strip():
        st.warning("Please paste some source code first.")
    else:
        with col2:
            with st.status("Processing...", expanded=True) as status:
                
                # Phase 1: Analysis & Detection
                status.write("🔍 Phase 1: Analyzing logic and detecting language...")
                analysis_result = analyze_and_detect(source_code)
                
                if analysis_result:
                    # Parse the analysis result
                    detected_lang = "Unknown"
                    summary_text = analysis_result
                    
                    if "DETECTED_LANGUAGE:" in analysis_result:
                        parts = analysis_result.split("SUMMARY:", 1)
                        if len(parts) > 0:
                            lang_line = parts[0].replace("DETECTED_LANGUAGE:", "").strip()
                            detected_lang = lang_line.split('\n')[0].strip() # Clean up any extra newlines
                        if len(parts) > 1:
                            summary_text = parts[1].strip()

                    # Phase 2: Logic Check for Mismatch
                    # We normalize to lower case for comparison (e.g., "Python" == "python")
                    if source_lang_selection != "Auto-Detect":
                        if detected_lang.lower() != source_lang_selection.lower():
                            warning_msg = (
                                f"**Notice:** User selected **{source_lang_selection}** as input "
                                f"but the code is in **{detected_lang}**. "
                                f"Showing the converted code in the selected output language."
                            )
                            warning_area.warning(warning_msg)

                    # Phase 3: Generation
                    status.write(f"🛠️ Phase 2: Generating {target_lang} code...")
                    raw_result = generate_target_code(source_code, summary_text, target_lang)
                    
                    if raw_result:
                        status.update(label="Conversion Complete!", state="complete", expanded=False)
                        
                        # Display Code
                        output_area.code(raw_result, language=target_lang.lower())
                        
                        # Download Button
                        st.download_button(
                            label=f"Download {target_lang} File",
                            data=raw_result,
                            file_name=f"converted_code_{target_lang}.txt",
                            mime="text/plain"
                        )
                    else:
                        status.update(label="Generation Failed", state="error")
                else:
                    status.update(label="Analysis Failed", state="error")