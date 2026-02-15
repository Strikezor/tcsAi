import streamlit as st
import os
import httpx
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import markdown
from xhtml2pdf import pisa
import json
from io import BytesIO
import random
import string

load_dotenv()

tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
KB_FILE_NAME = "synthetic_kb.json"
VECTOR_DB_DIR = "./chroma_db_kb"

client = httpx.Client(verify=False)

st.set_page_config(page_title="KnowledgeForge", layout="centered")
st.title("KnowledgeForge!")
st.markdown("Enter a bug description or error log below to generate a formal resolution markup based on the Synthetic KB.")

def get_general_resolution(query):
    # Agent 3: This agent will generate output if the knowledge base doesnt have this info
    
    fallback_llm = ChatOpenAI( 
        base_url="https://genailab.tcs.in", 
        model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=client
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a Senior Site Reliability Engineer. 
    The user has reported a bug that was **NOT found** in the internal Knowledge Base.
    Your task is to generate a **General Resolution Report** based on industry best practices and your general knowledge.

    User Bug Description: {input}

    Output Format (Markdown):
    # 🚨 Incident Resolution Report (General AI)
    **Source:** General Knowledge (Not from KB)
    **Severity:** [Estimated Severity]
    
    ## 📋 Incident Summary
    [Summarize the issue]

    ## 🔍 Likely Root Causes
    1. [Possible Cause 1]
    2. [Possible Cause 2]

    ## 🛠️ Recommended Resolution Steps
    1. [Step 1]
    2. [Step 2]
    
    ## ⚙️ Configuration To Check
    ```yaml
    [Relevant config snippets if applicable]
    ```
    
    """)

    chain = prompt | fallback_llm | StrOutputParser()
    return chain.invoke({"input": query})

def convert_to_pdf(markdown_content):
    """
    Converts Markdown text to a PDF byte stream.
    """
    # Convert Markdown to HTML
    html_text = markdown.markdown(markdown_content)
    
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Helvetica, sans-serif; font-size: 12px; }}
            h1 {{ color: #333; font-size: 18px; }}
            h2 {{ color: #555; font-size: 16px; margin-top: 10px; }}
            code {{ background-color: #f4f4f4; padding: 2px; font-family: Courier; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        {html_text}
    </body>
    </html>
    """
    
    # 3. Generate PDF
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(styled_html, dest=pdf_buffer)
    
    if pisa_status.err:
        return None
    
    return pdf_buffer.getvalue()

def validate_kb_entry_agent(entry_data):
    """
    Agent 1: Validates the input provided by user for kb generation
    Returns: (is_valid: bool, feedback: str)
    """
    # lightweight LLM for validation
    validator_llm = ChatOpenAI( 
        base_url="https://genailab.tcs.in", 
        model="azure/genailab-maas-gpt-4.1-nano", 
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=client,
        temperature=0.0 
    )

    validation_prompt = ChatPromptTemplate.from_template("""
    You are a Quality Assurance AI for a Site Reliability Engineering Knowledge Base.
    Your task is to validate the following draft KB entry provided by a user.

    Validation Rules:
    1. **Relevance:** The content must be related to software bugs, infrastructure, or technical errors.
    2. **Spam Check:** Reject nonsensical input, profanity, or testing strings (e.g., "asdf", "test").
    3. **Format Check:** Ignore the format in which user gave input; focus only on the content.
    4. **Grammar Check:** Ignore the grammatical mistakes provided by user in input.

    Draft Entry:
    {entry_json}

    Output Format:
    If Valid: Return exactly the string "VALID"
    If Invalid: Return string "INVALID: <brief explanation of what is wrong>"
    """)

    chain = validation_prompt | validator_llm | StrOutputParser()
    
    entry_str = json.dumps(entry_data, indent=2)
    
    try:
        result = chain.invoke({"entry_json": entry_str})
        if result.strip().upper() == "VALID":
            return True, "Valid"
        else:
            return False, result.replace("INVALID:", "").strip()
    except Exception as e:
        return False, f"Validation Agent Failed: {e}"

def refine_kb_entry_agent(entry_data):
    """
    Agent 2: Refines grammar AND infers missing technical details (Architecture, Tags, etc.)
    """
    refiner_llm = ChatOpenAI( 
        base_url="https://genailab.tcs.in", 
        model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
        api_key=os.getenv("OPENAI_API_KEY"),
        http_client=client,
        temperature=0.4
    )

    refine_prompt = ChatPromptTemplate.from_template("""
    You are a Senior Site Reliability Engineer. Your task is to polish the draft KB entry and AUTO-COMPLETE missing technical details.
    
    Instructions:
    1. **Polish:** Rewrite 'Summary', 'Root Cause', and 'Resolution' to be professional and concise.
    2. **Infer Category & Tags:** Based on the description, fill in the 'category' and generate 3-5 relevant 'tags'.
    3. **Infer Impact & Detection:** If missing, generate a plausible 'impact' (e.g., "Partial outage") and 'detection' method.
    4. **Generate Architecture:** If 'system_architecture' is empty, generate a plausible node/edge structure based on the product (e.g., "Client -> Load Balancer -> App -> DB").
    5. **Generate Limitations:** Provide 1-2 standard operational limitations for this type of fix.
    6. **Structure:** Return ONLY the valid JSON object.

    Draft Entry:
    {entry_json}
    """)

    chain = refine_prompt | refiner_llm | StrOutputParser()
    
    entry_str = json.dumps(entry_data, indent=2)
    
    try:
        result = chain.invoke({"entry_json": entry_str})
        
        # Clean up markdown
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        elif "```" in result:
            result = result.split("```")[0]
            
        return json.loads(result)
        
    except Exception as e:
        st.error(f"Refinement Agent Failed: {e}")
        return entry_data

@st.cache_resource(show_spinner=False)
def initialize_rag_system():
    """
    Loads the PDF, chunks it, embeds it, and prepares the RAG chain.
    Cached to prevent reloading on every interaction.
    """
    if not os.path.exists(KB_FILE_NAME):
        st.error(f"CRITICAL ERROR: '{KB_FILE_NAME}' not found in the directory. Please add the file to run the app.")
        return None

    with st.spinner(f"Ingesting {KB_FILE_NAME} into Vector Database..."):
        try:
            with open(KB_FILE_NAME, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_text = json.dumps(data, indent=2)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200, 
            chunk_overlap=300,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(raw_text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        embedding_model = OpenAIEmbeddings( 
            base_url="https://genailab.tcs.in", 
            model="azure/genailab-maas-text-embedding-3-large", 
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=client
        ) 

        vectordb = Chroma.from_documents(documents, embedding_model)
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatOpenAI( 
            base_url="https://genailab.tcs.in", 
            model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=client 
        )

        template = """
        You are a Senior Site Reliability Engineer. Your task is to analyze the user's reported bug 
        and map it to Known Error(s) in the Knowledge Base context provided.

        Instructions:
        1. Search the Context for the specific error, log message, or what is described.
        2. If the issue is linked to MULTIPLE KB articles in the context, synthesize the information into a SINGLE comprehensive report.
        3. Combine resolution steps and configuration changes from all relevant articles.
        4. If NOT found, state explicitly that no matching KB was found.
        5. Do NOT chat. Only output the structured response.

        Output Format (Markdown):
        # 🚨 Incident Resolution Report
        **KB Reference:** [Article ID 1], [Article ID 2], ... (List all relevant KB IDs separated by commas)
        **Severity:** [Highest Severity Level among the KBs]
        
        ## 📋 Incident Summary
        [Combined brief description of the issue based on the KBs]

        ## 🔍 Root Cause Analysis
        [Technical explanation of the root cause. If multiple causes exist, list them clearly.]

        ## 🛠️ Resolution Steps
        1. [Step 1]
        2. [Step 2]
        (Merge steps from all relevant articles)
        
        ## ⚙️ Configuration Changes
        ```yaml
        [Insert relevant config snippets from ALL relevant KBs here]
        ```

        ## Usage Mterics
        ```yaml
        [Show the usage metrics like views, helpful_votes, search hits]
        ```

        ## ⚠️ Limitations / Notes
        * [Limitation 1]
        * [Limitation 2]

        <context>
        {context}
        </context>

        User Bug Description: {input}
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    
# Logic
rag_chain = initialize_rag_system()

# Input Form
with st.form("bug_report_form"):
    user_input = st.text_area(
        "Bug Description / Log Output", 
        height=200, 
        placeholder="Example: Users are getting '415 Unsupported Media Type' when making POST requests to the API..."
    )
    
    submitted = st.form_submit_button("Generate Reports")


if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if submitted and user_input and rag_chain:
    st.session_state.last_query = user_input
    with st.spinner("Analyzing Knowledge Base..."):
        try:
            st.session_state.current_report = rag_chain.invoke(user_input)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.session_state.current_report:
    if "No matching KB was found" in st.session_state.current_report:
        st.warning("⚠️ No exact match found in the internal Knowledge Base.")
        
        st.markdown("Would you like to ask the General AI for a potential solution?")
        if st.button("🤖 Ask LLM (General Knowledge)"):
            with st.spinner("Consulting General AI for best practices..."):
                try:
                    fallback_response = get_general_resolution(st.session_state.last_query)
                    st.session_state.current_report = fallback_response
                    st.rerun() 
                except Exception as e:
                    st.error(f"Fallback failed: {e}")
    else:
        st.divider()
        st.subheader("Generated Resolution")
        st.markdown(st.session_state.current_report)
        
        # PDF Download
        pdf_bytes = convert_to_pdf(st.session_state.current_report)
        if pdf_bytes:
            st.download_button(
                label="📄 Download Report as PDF",
                data=pdf_bytes,
                file_name="incident_resolution.pdf",
                mime="application/pdf",
            )
        
        with st.expander("View Raw Markdown Source"):
            st.code(st.session_state.current_report, language="markdown")

elif submitted and not user_input:
    st.warning("Please enter a bug description.")


st.divider()
st.subheader("📝 Contribute to Knowledge Base")
st.caption("If the resolution wasn't found above, add a new known error to the database.")


with st.expander("➕ Open New KB Entry Form"):
    with st.form("add_kb_entry_form"):
        st.caption("Provide the core details, and our AI will generate the architecture, tags, and metadata for you.")
        
        new_title = st.text_input("KB Title", placeholder="e.g., Service X: Connection Timeout")
        
        col1, col2 = st.columns(2)
        with col1:
            new_product = st.text_input("Product Name", placeholder="e.g., Firefly III")
            new_severity = st.selectbox("Severity", ["SEV-1", "SEV-2", "SEV-3", "SEV-4"])
        with col2:
            new_environment = st.selectbox("Environment", ["prod", "staging", "dev"])
            new_status = st.selectbox("Status", ["Resolved", "Mitigated", "Known Issue"])

        new_summary = st.text_area("Incident Summary", placeholder="Briefly describe what went wrong...")
        
        col3, col4 = st.columns(2)
        with col3:
            new_root_cause = st.text_area("Root Cause", placeholder="Why did it happen?")
        with col4:
            new_resolution = st.text_area("Resolution Steps", placeholder="How do you fix it?")

        new_logs = st.text_area("Error Logs (Optional)", height=80, placeholder="Paste relevant error logs here...")
        new_config_raw = st.text_area("Config Changes (Optional - Format KEY: VALUE)", height=80)

        submit_new_kb = st.form_submit_button("✨ Generate & Save Entry")

    if submit_new_kb:
        if new_title and new_summary and new_resolution:
            try:
                kb_id = "KB-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                inc_id = "INC-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                
                config_dict = {}
                if new_config_raw:
                    for line in new_config_raw.split("\n"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            config_dict[k.strip()] = v.strip()

                new_entry = {
                    "kb_id": kb_id,
                    "title": new_title,
                    "metadata": {
                        "product": new_product,
                        "category": "", # Agent will fill
                        "severity": new_severity,
                        "environment": new_environment,
                        "status": new_status,
                        "created": "2026-02-14T12:00:00Z",
                        "updated": "2026-02-14T12:00:00Z",
                        "tags": [] # Agent will fill
                    },
                    "incident": {
                        "incident_id": inc_id,
                        "bug_id": "BUG-GEN",
                        "summary": new_summary,
                        "impact": "", # Agent will fill
                        "detection": "", # Agent will fill
                        "root_cause": new_root_cause,
                        "resolution": new_resolution
                    },
                    "resolution_logs": [l.strip() for l in new_logs.split("\n") if l.strip()],
                    "limitations": [], # Agent will fill
                    "config_changes": config_dict,
                    "system_architecture": {}, # Agent will fill
                    "usage_metrics": { "views": 0, "helpful_votes": 0, "not_helpful_votes": 0, "avg_time_on_page_sec": 0, "search_hits": 0 }
                }

                # agent pipeline
                
                # 1. Validate
                with st.spinner("🤖 Agent 1: Validating technical accuracy..."):
                    is_valid, validation_msg = validate_kb_entry_agent(new_entry)
            
                if not is_valid:
                    st.error(f"❌ Validation Failed: {validation_msg}")
                    st.stop() 

                st.success("✅ Input Validated")

                # 2. Refine & Auto-Complete
                with st.spinner("🧠 Agent 2: Inferring architecture, tags, and polishing content..."):
                    refined_entry = refine_kb_entry_agent(new_entry)
                
                with st.expander("✨ View Generated Entry"):
                    st.json(refined_entry)

                # 3. Save
                with st.spinner("💾 Saving to Knowledge Base..."):
                    current_data = []
                    if os.path.exists(KB_FILE_NAME):
                        with open(KB_FILE_NAME, "r", encoding="utf-8") as f:
                            current_data = json.load(f)
                    
                    current_data.append(refined_entry)
                    
                    with open(KB_FILE_NAME, "w", encoding="utf-8") as f:
                        json.dump(current_data, f, indent=2)
                    
                    st.cache_resource.clear()
                    st.success(f"🎉 Saved **{refined_entry['kb_id']}** to the database!")
                
            except Exception as e:
                st.error(f"Failed to save entry: {e}")
        else:
            st.warning("Please fill in Title, Summary, Root Cause, and Resolution.")


with st.expander("📂 Upload KB File (JSON)"):
    st.caption("Upload a JSON file containing KB entries. Each entry will be validated and refined before insertion.")
    
    uploaded_kb_file = st.file_uploader("Choose a JSON file", type=["json"])
    
    if uploaded_kb_file is not None:
        if st.button("🚀 Process & Merge Uploaded File"):
            try:
                uploaded_data = json.load(uploaded_kb_file)
                
                if isinstance(uploaded_data, dict):
                    uploaded_data = [uploaded_data]
                
                if not isinstance(uploaded_data, list):
                    st.error("Invalid format: Root element must be a list of KB entries.")
                    st.stop()

                success_count = 0
                error_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                current_kb_data = []
                if os.path.exists(KB_FILE_NAME):
                    with open(KB_FILE_NAME, "r", encoding="utf-8") as f:
                        current_kb_data = json.load(f)

                for i, entry in enumerate(uploaded_data):
                    status_text.text(f"Processing entry {i+1}/{len(uploaded_data)}...")
                    
                    is_valid, msg = validate_kb_entry_agent(entry)
                    
                    if is_valid:
                        refined_entry = refine_kb_entry_agent(entry)
                        
                        if "kb_id" not in refined_entry or not refined_entry["kb_id"]:
                             refined_entry["kb_id"] = "KB-" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

                        current_kb_data.append(refined_entry)
                        success_count += 1
                    else:
                        st.warning(f"Skipped Entry {i+1}: {msg}")
                        error_count += 1
                    
                    progress_bar.progress((i + 1) / len(uploaded_data))

                with open(KB_FILE_NAME, "w", encoding="utf-8") as f:
                    json.dump(current_kb_data, f, indent=2)
                
                st.cache_resource.clear()
                status_text.text("Done!")
                st.success(f"Processing Complete! ✅ Merged {success_count} entries. ❌ Skipped {error_count} entries.")
                
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please check the syntax.")
            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")