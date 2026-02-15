import streamlit as st
import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration & Setup ---
load_dotenv()

# Set page layout to wide immediately
st.set_page_config(
    page_title="Legacy Code Converter",
    layout="wide",
    page_icon="🔄"
)

# --- specific API Client Setup (Strict Adherence) ---
try:
    # HTTP client with verification disabled
    client_httpx = httpx.Client(verify=False)

    # OpenAI-compatible client configuration
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

def analyze_code(source_code, source_lang):
    """
    Step 1: Analyzes the code to understand logic and business rules.
    Acts as the 'retrieval' phase of our logic.
    """
    system_prompt = (
        "You are an expert Senior Software Architect specializing in legacy systems. "
        "Your goal is to analyze the provided code deeply. "
        "Identify the core logic, business rules, variable transformations, and control flow. "
        "Do not generate code yet. Output a concise technical summary of what this code does."
    )
    
    user_prompt = f"Source Language: {source_lang}\n\nCode:\n{source_code}"

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2 
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during Analysis phase: {e}")
        return None

def generate_target_code(source_code, analysis_summary, source_lang, target_lang):
    """
    Step 2: Generates the target code using the original source AND the analysis summary.
    """
    system_prompt = (
        f"You are an expert Polyglot Developer. Convert the following {source_lang} code to {target_lang}. "
        "Use the provided technical analysis to ensure logical equivalence and accuracy. "
        "IMPORTANT: Output ONLY the raw code. Do not include markdown formatting (like ```python), "
        "explanations, or conversational text. Just the executable code."
    )

    user_prompt = (
        f"Technical Analysis/Context:\n{analysis_summary}\n\n"
        f"Original Source Code:\n{source_code}"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # Low temperature for precision
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error during Generation phase: {e}")
        return None

def clean_output(text):
    """
    Removes markdown code fences if the model accidentally includes them.
    """
    if not text:
        return ""
    lines = text.split('\n')
    # Remove start/end code fences if present
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)

# --- UI Layout ---

st.title("🔄 Legacy Code Converter")
st.markdown("### Intelligent Two-Pass Conversion (Analysis $\\to$ Generation)")
st.markdown("---")

# Layout: Two Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Source Code")
    source_lang = st.selectbox(
        "Source Language",
        ["Auto-Detect", "COBOL", "Fortran", "C++", "Java", "Pascal", "VB.NET", "Perl"],
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
    
    # Placeholder for the output logic to run
    convert_btn = st.button("Convert Code", type="primary", use_container_width=True)
    
    output_area = st.empty() # Placeholder for dynamic updates

# --- Main Logic ---

if convert_btn:
    if not source_code.strip():
        st.warning("Please paste some source code first.")
    else:
        with col2:
            with st.status("Processing...", expanded=True) as status:
                
                # Phase 1: Analysis
                status.write("🔍 Phase 1: Analyzing logic and business rules...")
                analysis = analyze_code(source_code, source_lang)
                
                if analysis:
                    # Optional: Show the analysis in an expander for transparency
                    with st.expander("View Logic Analysis (Intermediate Step)"):
                        st.write(analysis)
                    
                    # Phase 2: Generation
                    status.write(f"🛠️ Phase 2: Generating {target_lang} code...")
                    raw_result = generate_target_code(source_code, analysis, source_lang, target_lang)
                    
                    if raw_result:
                        final_code = clean_output(raw_result)
                        status.update(label="Conversion Complete!", state="complete", expanded=False)
                        
                        # Display Code
                        output_area.code(final_code, language=target_lang.lower())
                        
                        # Download Button
                        file_ext = {
                            "Python": "py", "Java": "java", "Go": "go", 
                            "Rust": "rs", "JavaScript": "js", "C#": "cs"
                        }.get(target_lang, "txt")
                        
                        st.download_button(
                            label=f"Download {target_lang} File",
                            data=final_code,
                            file_name=f"converted_code.{file_ext}",
                            mime="text/plain"
                        )
                    else:
                        status.update(label="Generation Failed", state="error")
                else:
                    status.update(label="Analysis Failed", state="error")