import os
import tempfile
import streamlit as st
import pandas as pd
import httpx
from typing import TypedDict, Literal, List, Annotated
import operator
from dotenv import load_dotenv
import re

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

tiktoken_cache_dir="./token"
os.environ["TIKTOKEN_CACHE_DIR"]=tiktoken_cache_dir
client = httpx.Client(verify=False) 

load_dotenv()

# 1. Configure Embeddings
embeddings = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"), # Fallback if not set
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

# --- State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    csv_data_exists: bool
    routing_decision: str
    context: str  # To store retrieved RAG context

# --- Node Logic ---

def router_node(state: AgentState):
    """
    Node 1: Guardrail & Router
    Classifies user intent: BLOCKED, GENERAL, or DATA_RAG.
    """
    messages = state["messages"]
    user_query = messages[-1].content
    csv_exists = state.get("csv_data_exists", False)

    system_prompt = (
        "You are a routing assistant for an Financial Advisor app. "
        "Classify the user's query into exactly one of these three categories:\n\n"
        "1. BLOCKED: The query is NOT related to finance, investment, math, or data analysis (e.g., cooking, politics, sports).\n"
        "2. DATA_RAG: The query requires analyzing specific uploaded data/portfolio (e.g., 'What is my total balance?', 'highest return', 'filter by location'). "
        "IMPORTANT: You can only choose DATA_RAG if the user has uploaded a CSV (Context: CSV Uploaded = {csv_status}). "
        "If they ask a data question but CSV Uploaded is False, route to GENERAL so the model can explain it needs data.\n"
        "3. GENERAL: The query is a conceptual financial question (e.g., 'What is a mutual fund?', 'Formula for ROI') or a greeting, OR a data request when no data is present.\n\n"
        "Return ONLY the category name."
        
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.format(csv_status=str(csv_exists))),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"query": user_query}).strip()
    
    # Fallback cleanup just in case
    if "BLOCKED" in decision: decision = "BLOCKED"
    elif "DATA_RAG" in decision: decision = "DATA_RAG"
    else: decision = "GENERAL"
    
    return {"routing_decision": decision}

def general_chat_node(state: AgentState):
    """
    Node 2: General Chat
    Handles conceptual queries without RAG.
    """
    messages = state["messages"]
    
    system_msg = (
        "You are a helpful Senior Financial Advisor. "
        "Answer the user's question clearly and concisely. "
        "Respond only with data relevant to Indian stock market and currency."
        "Do not make up specific user data; stick to general financial concepts, math, and definitions."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        *messages
    ])
    
    chain = prompt | llm
    response = chain.invoke({})
    return {"messages": [response]}

def data_analysis_node(state: AgentState):
    """
    Node 3: Data Analysis (RAG)
    Handles queries using the vector store context.
    """
    messages = state["messages"]
    user_query = messages[-1].content
    
    # Retrieve relevant documents
    # Note: 'retriever' is injected via st.session_state globally for this demo
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
    workflow.add_node("blocked", blocked_node)

    # set entry point
    workflow.set_entry_point("router")

    # Conditional Edges based on router output
    def route_step(state):
        decision = state["routing_decision"]
        if decision == "BLOCKED":
            return "blocked"
        elif decision == "DATA_RAG":
            return "rag"
        else:
            return "general"

    workflow.add_conditional_edges(
        "router",
        route_step,
        {
            "blocked": "blocked",
            "rag": "rag",
            "general": "general"
        }
    )

    # Edges to END
    workflow.add_edge("general", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("blocked", END)

    return workflow.compile()

# --- Streamlit UI ---

st.set_page_config(page_title="Smart Financial Advisor", page_icon="📈")

st.title("📈 AI Financial Advisor & Portfolio Analyzer")
st.markdown("Powered by **DeepSeek-V3** & **LangGraph**")

# Sidebar
with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Investment CSV", type=["csv"])
    
    # Check if file is uploaded and not yet processed
    if uploaded_file and ("csv_processed" not in st.session_state or not st.session_state.csv_processed):
        with st.spinner("Indexing Financial Data..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            loader = CSVLoader(file_path=tmp_path)
            data = loader.load()
            vectorstore = Chroma.from_documents(documents=data, embedding=embeddings)
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
            st.session_state.csv_processed = True
            os.remove(tmp_path)
            st.success("Portfolio Indexed Successfully!")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat Logic
if prompt := st.chat_input("Ask about finance or your portfolio..."):
    # 1. Add User Message to History
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("human"):
        st.markdown(prompt)

    # 2. Prepare Graph Input
    app = build_graph()
    initial_state = {
        "messages": st.session_state.messages,
        "csv_data_exists": st.session_state.get("csv_processed", False),
        "routing_decision": "",
        "context": ""
    }

    # 3. Run Graph
    with st.chat_message("ai"):
        with st.status("Processing Request...", expanded=True) as status:
            st.write("1️⃣  Analyzing intent with Router...")
            final_state = app.invoke(initial_state)
            
            decision = final_state.get("routing_decision", "Unknown")
            
            # Update Dropdown Content with Steps
            st.write(f"2️⃣  Intent classified as: **{decision}**")
            
            if decision == "BLOCKED":
                st.write("3️⃣  Action: Blocking off-topic request.")
                final_label = "🛑 Mode: Blocked (Off-Topic)"
                state_color = "error"
            elif decision == "DATA_RAG":
                st.write("3️⃣  Action: Retrieving relevant rows from CSV...")
                st.write("4️⃣  Action: Generating analysis...")
                final_label = "📊 Mode: RAG (Portfolio Data)"
                state_color = "complete"
            else:
                st.write("3️⃣  Action: querying internal financial knowledge base...")
                final_label = "🧠 Mode: General Knowledge"
                state_color = "complete"

            # Update the Dropdown Label (Header)
            status.update(label=final_label, state=state_color, expanded=False)
        # 4. Display Response
        ai_response = final_state["messages"][-1]
        st.markdown(ai_response.content)
        
        # Update Session State
        st.session_state.messages.append(ai_response)