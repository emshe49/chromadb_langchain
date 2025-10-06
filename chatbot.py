"""
UET Mardan Web RAG (Retrieval-Augmented Generation)
- Loads department content from the UET Mardan website
- Fixes encoding issues (UTF-8)
- Cleans the extracted text for better QA
- Uses Hugging Face Embeddings + QA (CPU only)
"""

# ============================================================
# 1️⃣ Imports & Environment Setup
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["USER_AGENT"] = "MyLangChainBot/1.0 (contact: youremail@example.com)"

# ============================================================
# 2️⃣ Web Data Loader (UET Mardan)
# ============================================================

from langchain_community.document_loaders import WebBaseLoader
import bs4

print("🌐 Loading data from UET Mardan website...")

loader = WebBaseLoader(
    web_paths=("https://www.uetmardan.edu.pk/uetm/Department/softwaredept",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("course-details-inner", "sidebar-title", "footer-about", "text")
        )
    ),
)

# Load the data
text_documents = loader.load()

# Fix encoding issues and clean text
for doc in text_documents:
    # Force UTF-8 decoding and clean invalid characters
    doc.page_content = (
        doc.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")
        .replace("\xa0", " ")
        .replace("â€”", "—")
        .replace("â€“", "-")
        .replace("â€˜", "'")
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€¦", "...")
        .strip()
    )

print("✅ Loaded and cleaned text documents:", len(text_documents))
print(text_documents[0].page_content)

# ============================================================
# 3️⃣ Text Chunking
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(text_documents)
print("✅ Number of chunks:", len(documents))

# ============================================================
# 4️⃣ Create Embeddings + Vector Store (Chroma)
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(documents, embedding_model)
print("✅ Chroma vector database created with website content.")

# ============================================================
# 5️⃣ QA Model (Hugging Face - Local)
# ============================================================

from transformers import pipeline

print("\n🤖 Loading QA model (deepset/roberta-base-squad2)...")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ============================================================
# 6️⃣ Helper Function for Clean Answers
# ============================================================

def ask_question(question: str):
    """Retrieve relevant context and generate a local QA answer."""
    results = db.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in results])

    answer = qa_pipeline({"question": question, "context": context})
    cleaned_answer = answer["answer"].replace("â€”", "—").replace("â€¦", "...").strip()

    print(f"\n💬 Question: {question}")
    print(f"🧠 Answer: {cleaned_answer}")
    print(f"✅ Confidence: {round(answer['score'] * 100, 2)}%\n")


# ============================================================
# 7️⃣ Initial Test Query
# ============================================================

ask_question("Who is the head of the Software Engineering Department?")
ask_question("What programs are offered by the Software Engineering Department?")
ask_question("What is the contact information of the department?")

# ============================================================
# 8️⃣ Interactive Q&A Chat
# ============================================================

print("\n✨ Ask me anything about the UET Software Engineering Department! (type 'exit' to quit)\n")

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    ask_question(user_query)
