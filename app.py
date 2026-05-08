import streamlit as st
from realtime_rag import search_and_scrape, build_vector_store, generate_answer

st.set_page_config(page_title="Industry Trend Analyzer", page_icon="📊", layout="wide")

st.title("📊 Automated Industry Trend Analyzer")
st.markdown("""
Welcome to the Real-Time RAG Analyzer. This tool bypasses standard LLM hallucinations by dynamically searching 
only **top-tier financial and consulting domains** (McKinsey, Reuters, Bloomberg, WSJ, etc.) to ground its answers.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask about recent market trends or earnings..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[:-1]])
        
        with st.status("🔍 Scanning premium industry sources...", expanded=True) as status:
            st.write("Extracting web data...")
            scraped_info = search_and_scrape(user_query, num_results=5)
            
            if not scraped_info:
                status.update(label="Failed to find data", state="error")
                st.error("Could not find relevant industry reports for this query.")
                st.stop()
                
            st.write("Chunking and vectorizing reports...")
            faiss_db = build_vector_store(scraped_info)
            
            st.write("Generating grounded analysis...")
            final_answer = generate_answer(user_query, faiss_db, chat_history=history_str)
            status.update(label="Analysis Complete", state="complete", expanded=False)
            
        st.markdown("### Analysis")
        st.write(final_answer)
        
        with st.expander("📚 View Cited Industry Sources"):
            for item in scraped_info:
                st.write(f"- {item['url']}")
                
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
