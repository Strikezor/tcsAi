from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv
import os
import httpx  


load_dotenv()
client = httpx.Client(verify=False)
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=client
) 
response = llm.invoke("Hi") 
print(response)