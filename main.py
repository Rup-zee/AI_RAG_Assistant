import os
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.title("AI RAG Assistant")

# Sidebar - URL Input
st.sidebar.header("URLs to Scrape")
urls_input = st.sidebar.text_area(
    "Enter URLs (one per line)", 
    value="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://www.ibm.com/topics/artificial-intelligence"
)
urls = [u.strip() for u in urls_input.split("\n") if u.strip()]

# Function to scrape pages
def scrape(url):
    st.info(f"Scraping: {url}")
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

# Build Vector DB
if st.sidebar.button("Build Vector DB"):
    with st.spinner("Scraping and building vector DB..."):
        documents = [scrape(url) for url in urls]
        full_text = "\n".join(documents)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_text(full_text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="vector_db")
        vector_db.persist()
        st.success("Vector database built and saved!")

# Load Vector DB
vector_db = Chroma(persist_directory="vector_db", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# Initialize Groq LLM
llm = ChatGroq(model="groq/compound-mini", groq_api_key=GROQ_API_KEY)

prompt = PromptTemplate.from_template("""
You are an intelligent RAG assistant. Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

Answer in clear, detailed paragraphs:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_pipeline = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# User Query
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Generating answer..."):
        response = rag_pipeline.invoke(query)
        st.subheader("RAG Answer")
        st.write(response.content)
