# Automated-Industry-Trend-Analyzer

# 📊 Real-Time Industry RAG Analyzer

An end-to-end Retrieval-Augmented Generation (RAG) web application designed to analyze real-time industry trends and financial data by strictly querying top-tier consulting and financial domains.

## 🚀 Project Overview
Traditional LLMs suffer from knowledge cutoffs and hallucinations when analyzing current market trends. This project solves that by dynamically scraping live data from verified sources (McKinsey, Reuters, Bloomberg, WSJ), vectorizing the unstructured HTML, and grounding the LLaMA-3.1 generation process in mathematical and factual context.

### Features
- **Real-Time Web Scraping:** Bypasses stale data by scraping live articles using the DuckDuckGo API and BeautifulSoup.
- **Strict Source Grounding:** Domain-restricted search ensures only premium data is vectorized.
- **In-Memory Vectorization:** Uses HuggingFace `MiniLM-L6-v2` embeddings and FAISS for fast similarity search.
- **Conversational Memory:** Maintains context across user queries using Streamlit Session State.
- **Hallucination Prevention:** Custom prompt engineering guarantees the LLM outputs only data present in the context.

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **Data Ingestion:** `requests`, `BeautifulSoup4`, `ddgs`
- **RAG Architecture:** LangChain, FAISS (Vector DB)
- **Embeddings:** HuggingFace `sentence-transformers`
- **LLM:** Meta LLaMA-3.1 (via Groq API)

## 💻 Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/industry-rag-analyzer.git
   cd industry-rag-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq API Key:
   Create a `.env` file in the root directory and add:
   ```env
   GROQ_API_KEY="your_api_key_here"
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```
