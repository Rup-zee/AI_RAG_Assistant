import os
import requests
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ------------------- 1. Load API Key -------------------

os.environ["GROQ_API_KEY"] = "Enter_your_API_key"

print("KEY LOADED:", os.getenv("GROQ_API_KEY") is not None)

# ------------------- 2. Web Scraping -------------------
def scrape(url):
    print(f"Scraping: {url}")
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://www.ibm.com/topics/artificial-intelligence",
    "https://cloud.google.com/learn/what-is-artificial-intelligence",
    "https://www.britannica.com/technology/artificial-intelligence"
    
]

documents = [scrape(url) for url in urls]
full_text = "\n".join(documents)

# ------------------- 3. Text Splitting -------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_text(full_text)

# ------------------- 4. Embeddings + VectorDB -------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_texts(chunks, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ------------------- 5. LLM (Groq) -------------------
llm = ChatGroq(
    model="groq/compound",
    groq_api_key=os.getenv("GROQ_API_KEY")
)
print(os.getenv("GROQ_API_KEY"))


# ------------------- 6. RAG Prompt -------------------
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

# ------------------- 7. Interactive Loop -------------------
print("\n===== RAG INTERACTIVE MODE =====\n")
print("Type 'exit' to quit.\n")

while True:
    query = input("Your question: ")
    if query.strip().lower() in ["exit", "quit"]:
        print("Exiting RAG assistant. Goodbye!")
        break

    response = rag_pipeline.invoke(query)
    print("\n===== RAG ANSWER =====\n")
    print(response.content)
    print("\n" + "-"*80 + "\n")
