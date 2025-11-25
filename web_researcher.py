from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def scrape_website(url):
    """Simple web scraper"""
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ")
    except Exception as e:
        return f"Error scraping {url}: {e}"

summary_prompt = PromptTemplate.from_template("""
You are an AI research assistant.
Summarize the following text into a clean, crisp, factual report:

TEXT:
{content}

SUMMARY:
""")

def summarize_text(text):
    chain = summary_prompt | model
    return chain.invoke({"content": text}).content

if __name__ == "__main__":
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.ibm.com/topics/artificial-intelligence"
    ]

    all_text = ""
    for url in urls:
        print(f"Scraping: {url}")
        all_text += scrape_website(url)[:5000]  # limit to prevent overload

    print("\nGenerating Summary...\n")
    summary = summarize_text(all_text)

    print("===== AI SUMMARY REPORT =====")
    print(summary)
