import os
import warnings
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load API keys securely from .env file
load_dotenv()
warnings.filterwarnings('ignore', category=DeprecationWarning)

def search_and_scrape(query, num_results=5):
    print(f"🔍 Searching the web for: '{query}'...")
    ddgs = DDGS()
    
    try:
    # Removed strict paywall domains to prevent scraper blocking
    results = list(ddgs.text(query, max_results=num_results))
    except Exception:
        return []
    
    scraped_data = []
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for res in results:
        url = res['href']
        try:
            response = requests.get(url, timeout=5, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([p.text.strip() for p in soup.find_all('p')])
            
            if len(text) > 300: 
                scraped_data.append({"url": url, "text": text})
        except Exception:
            pass
            
    return scraped_data

def build_vector_store(scraped_data):
    if not scraped_data:
        raise ValueError("No valid industry data was scraped. Try rephrasing.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    chunks = []
    metadatas = []
    
    for data in scraped_data:
        splits = text_splitter.split_text(data["text"])
        chunks.extend(splits)
        metadatas.extend([{"url": data["url"]}] * len(splits))
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)

def generate_answer(query, vector_store, chat_history=""):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)
    
    context = "\n\n".join([f"Source URL: {doc.metadata['url']}\nContent: {doc.page_content}" for doc in relevant_docs])
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
    
    prompt = f"""
    You are an elite Industry Trends Analyst. Answer the user's question based strictly on the provided Context.
    If the context does not contain the answer, say "I cannot answer this based on the retrieved industry reports."
    
    Previous Conversation:
    {chat_history}
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    return llm.invoke(prompt).content
