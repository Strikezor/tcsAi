import streamlit as st 
from pdfminer.high_level import extract_text 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile 
import os 
import httpx 
from dotenv import load_dotenv
import certifi

tiktoken_cache_dir="./token"
os.environ["TIKTOKEN_CACHE_DIR"]=tiktoken_cache_dir
client = httpx.Client(verify=False) 

load_dotenv()

# Client setup to bypass SSL verification if needed
client = httpx.Client(verify=False)

# LLM and Embedding setup 
llm = ChatOpenAI( 
    base_url="https://genailab.tcs.in", 
    model="azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=client 
) 

embedding_model = OpenAIEmbeddings( 
    base_url="https://genailab.tcs.in", 
    model="azure/genailab-maas-text-embedding-3-large", 
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=client
) 

st.set_page_config(page_title="RAG PDF Summarizer") 
st.title("RAG-powered PDF Summarizer") 

upload_file = st.file_uploader("Upload a PDF", type="pdf") 

if upload_file: 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file: 
        temp_file.write(upload_file.read()) 
        temp_file_path = temp_file.name 
    
    # Step 1: Extract text 
    raw_text = extract_text(temp_file_path) 
    
    # Step 2: Chunking 

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len, # Uses character count instead of downloading tiktoken
    is_separator_regex=False
    )
    chunks = text_splitter.split_text(raw_text) 

    
    
    # Step 3: Embed and store in Chroma
    with st.spinner("Indexing document..."): 
        vectordb = Chroma.from_texts(
            chunks, 
            embedding_model, 
            persist_directory="./chroma_index"
        ) 
    
    # Step 4: RAG QA Chain (Updated)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}) 
    
    # Define the prompt for the "stuff" chain
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    # Create the modern chain structure
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    # Step 5: Ask summarization prompt 
    summary_prompt = "Please summarize this document based on the key topics:" 
    with st.spinner("Running RAG summarization..."): 
        # result = rag_chain.invoke(summary_prompt) # Old way
        result = rag_chain.invoke({"input": summary_prompt}) # New way
    
    st.subheader("Summary") 
    # Extracting the result text - key is now "answer" instead of "result"
    st.write(result["answer"])
    
    # Optional: Clean up temp file after processing
    os.remove(temp_file_path)