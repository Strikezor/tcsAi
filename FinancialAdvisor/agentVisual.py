import os
import re
import tempfile
import streamlit as st
import pandas as pd
import httpx
import plotly.express as px
from typing import TypedDict, Literal, List, Annotated
import operator
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

# --- Configuration & Setup ---

tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
client = httpx.Client(verify=False)

load_dotenv()

# 1. Configure Embeddings
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=client
)

# 2. Configure LLM
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=client,
    temperature=0
)

# --- Helper Functions ---

def execute_visualization_code(code_string: str, df: pd.DataFrame):
    """
    Executes the generated Python code to create a Plotly figure.
    """
    local_scope = {'df': df, 'px': px}
    try:
        # Execute the code in a restricted scope
        exec(code_string, {}, local_scope)
        # Extract the figure
        return local_scope.get('fig', None)
    except Exception as e:
        st.error(f"Visualization Error: {e}")
        return None

# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    csv_data_exists: bool
    routing_decision: str
    context: str
    visualization_code: str # New field to store generated plotting code

# --- Node Logic ---

def router_node(state: AgentState):
    """
    Node 1: Guardrail & Router
    Classifies intent: BLOCKED, GENERAL, DATA_RAG, or VISUALIZATION.
    """
    messages = state["messages"]
    user_query = messages[-1].content
    csv_exists = state.get("csv_data_exists", False)

    system_prompt = (
        "You are a routing assistant for a Financial Advisor app. "
        "Classify the user's query into exactly one of these four categories:\n\n"
        "1. BLOCKED: The query is NOT related to finance, investment, math, or data analysis.\n"
        "2. VISUALIZATION: The user explicitly asks to 'plot', 'graph', 'chart', 'visualize', or show a 'trend' of their data. "
        "IMPORTANT: Only choose VISUALIZATION if CSV Uploaded = True. If False, route to GENERAL.\n"
        "3. DATA_RAG: The query requires analyzing specific uploaded data (e.g., 'What is my total balance?', 'highest return') but DOES NOT ask for a chart.\n"
        "4. GENERAL: Conceptual questions, greetings, or data requests when no CSV is present.\n\n"
        "Context: CSV Uploaded = {csv_status}\n"
        "Return ONLY the category name."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(csv_status=str(csv_exists))),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"query": user_query}).strip()
    
    # Clean up response
    if "BLOCKED" in decision: decision = "BLOCKED"
    elif "DATA_RAG" in decision: decision = "DATA_RAG"
    elif "VISUALIZATION" in decision: decision = "VISUALIZATION"
    else: decision = "GENERAL"
    
    return {"routing_decision": decision}

def visualization_generator_node(state: AgentState):
    """
    New Node: Visualization Generator
    Generates Python code for Plotly Express based on DF schema.
    """
    messages = state["messages"]
    user_query = messages[-1].content
    
    # Retrieve dataframe schema from session state
    if "df" in st.session_state:
        df_dtypes = str(st.session_state.df.dtypes)
    else:
        df_dtypes = "Dataframe not found."

    system_msg = (
        "You are a Python Data Engineer specializing in Plotly Express. "
        "Generate valid Python code to create a chart based on the user's request and the dataframe schema provided below.\n\n"
        f"Dataframe Schema (df.dtypes):\n{df_dtypes}\n\n"
        "Strict Constraints:\n"
        "1. Assume the pandas DataFrame is already loaded in a variable named 'df'.\n"
        "2. Do NOT print anything.\n"
        "3. Assign the final Plotly figure object to a variable named 'fig'.\n"
        "4. Use 'plotly.express' imported as 'px'.\n"
        "5. Do NOT wrap the code in markdown fences (like ```python). Return raw code only.\n"
        "6. Handle empty data or errors by setting 'fig = None'.\n"
        "7. Create a title for the chart based on the query."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", user_query)
    ])
    
    chain = prompt | llm | StrOutputParser()
    code = chain.invoke({"query": user_query})
    
    # Clean up potential markdown formatting if the model slips
    code = code.replace("```python", "").replace("```", "").strip()
    
    return {
        "visualization_code": code, 
        "messages": [AIMessage(content="I have generated the visualization for you.")]
    }

def general_chat_node(state: AgentState):
    """
    Node 2: General Chat
    """
    messages = state["messages"]
    system_msg = (
        "You are a helpful Senior Financial Advisor. "
        "Answer the user's question clearly and concisely. "
        "Respond only with data relevant to Indian stock market and currency."
        "Do not make up specific user data; stick to general financial concepts, math, and definitions."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_msg), *messages])
    chain = prompt | llm
    response = chain.invoke({})
    return {"messages": [response]}

def data_analysis_node(state: AgentState):
    """
    Node 3: Data Analysis (RAG)
    """
    messages = state["messages"]
    user_query = messages[-1].content
    
    docs = []
    if "retriever" in st.session_state and st.session_state.retriever:
        docs = st.session_state.retriever.invoke(user_query)
    
    context_text = "\n\n".join([d.page_content for d in docs])
    
    system_msg = (
        "You are a Smart Portfolio Analyzer. "
        "Use the following retrieved data from the user's CSV file to answer the question.\n\n"
        f"Context Data:\n{context_text}\n\n"
        "If the answer is not in the context, state that you cannot find that information in the uploaded file."
        "Perform any necessary math based on the data provided."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", user_query)
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    return {"messages": [response], "context": context_text}

def blocked_node(state: AgentState):
    """
    Node 4: Guardrail Rejection
    """
    msg = AIMessage(content="I am a financial advisor only. I cannot assist with non-financial topics like cooking, politics, or general lifestyle queries.")
    return {"messages": [msg]}

# --- Graph Construction ---

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("general", general_chat_node)
    workflow.add_node("rag", data_analysis_node)
    workflow.add_node("visualizer", visualization_generator_node) # New Node
    workflow.add_node("blocked", blocked_node)

    # Set Entry Point
    workflow.set_entry_point("router")

    # Routing Logic
    def route_step(state):
        decision = state["routing_decision"]
        if decision == "BLOCKED": return "blocked"
        elif decision == "VISUALIZATION": return "visualizer"
        elif decision == "DATA_RAG": return "rag"
        else: return "general"

    workflow.add_conditional_edges(
        "router",
        route_step,
        {
            "blocked": "blocked",
            "rag": "rag",
            "visualizer": "visualizer",
            "general": "general"
        }
    )

    # Edges to END
    workflow.add_edge("general", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("visualizer", END)
    workflow.add_edge("blocked", END)

    return workflow.compile()

# --- Streamlit UI ---

st.set_page_config(page_title="Smart Financial Advisor", page_icon="📈")
st.title("📈 AI Financial Advisor & Portfolio Analyzer")

# Sidebar
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Investment CSV", type=["csv"])
    
    if uploaded_file and ("csv_processed" not in st.session_state or not st.session_state.csv_processed):
        with st.spinner("Indexing Financial Data..."):
            try:
                # 1. Read and Fix Content
                file_content = uploaded_file.getvalue().decode("utf-8")
                fixed_content = re.sub(r' (INV\d+)', r'\n\1', file_content)

                # 2. Save to Temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as tmp:
                    tmp.write(fixed_content)
                    tmp_path = tmp.name

                # 3. Load Data for RAG
                loader = CSVLoader(file_path=tmp_path)
                data = loader.load()
                
                # 4. Load Data for Visualization (Pandas)
                # We read the cleaned temp file into a DataFrame
                df = pd.read_csv(tmp_path)
                st.session_state.df = df # Store for Visualization Node

                # 5. Filter Empty Documents & Index
                data = [doc for doc in data if doc.page_content and doc.page_content.strip()]
                
                if not data:
                    st.error("CSV file is empty or contains no readable text data.")
                else:
                    vectorstore = Chroma.from_documents(documents=data, embedding=embeddings)
                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
                    st.session_state.csv_processed = True
                    st.success(f"Portfolio Indexed Successfully! ({len(data)} rows)")
            
            except Exception as e:
                st.error(f"Error processing CSV: {e}")
            finally:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("Ask about finance, portfolio, or request a chart..."):
    # User Message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    # Agent Execution
    app = build_graph()
    initial_state = {
        "messages": st.session_state.messages,
        "csv_data_exists": st.session_state.get("csv_processed", False),
        "routing_decision": "",
        "context": "",
        "visualization_code": ""
    }

    with st.chat_message("ai"):
        with st.status("Thinking...", expanded=True) as status:
            st.write("1️⃣  Analyzing intent with Router...")
            final_state = app.invoke(initial_state)
            
            decision = final_state.get("routing_decision", "Unknown")
            st.write(f"2️⃣  Intent classified as: **{decision}**")
            
            if decision == "BLOCKED":
                st.write("3️⃣  Action: Blocking off-topic request.")
                final_label = "🛑 Mode: Blocked (Off-Topic)"
                state_color = "error"
            
            elif decision == "VISUALIZATION":
                st.write("3️⃣  Action: Analyzing DataFrame Schema...")
                st.write("4️⃣  Action: Generating Plotly Code...")
                final_label = "🎨 Mode: Visualization Engine"
                state_color = "complete"
            
            elif decision == "DATA_RAG":
                st.write("3️⃣  Action: Retrieving relevant rows from CSV...")
                st.write("4️⃣  Action: Generating analysis...")
                final_label = "📊 Mode: RAG (Portfolio Data)"
                state_color = "complete"
            
            else:
                st.write("3️⃣  Action: Querying internal financial knowledge base...")
                final_label = "🧠 Mode: General Knowledge"
                state_color = "complete"

            status.update(label=final_label, state=state_color, expanded=False)
        
        # Display Final Response
        ai_response = final_state["messages"][-1]
        st.markdown(ai_response.content)
        st.session_state.messages.append(ai_response)

        # Check for Visualization Code and Execute
        if final_state.get("visualization_code"):
            with st.spinner("Rendering Chart..."):
                code = final_state["visualization_code"]
                # We use the df stored in session state
                if "df" in st.session_state:
                    fig = execute_visualization_code(code, st.session_state.df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Could not generate chart. Please check data or try a different query.")
                else:
                    st.error("Dataframe not available for visualization.")