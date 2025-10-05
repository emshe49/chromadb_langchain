"""
Simplified RAG (Retrieval-Augmented Generation) Script
- Uses Hugging Face Embeddings (CPU friendly)
- Uses Hugging Face QA Model (no OpenAI)
- Fixed USER_AGENT warning
"""

# ============================================================
# 1️⃣ Imports & Environment Setup
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

# Prevent USER_AGENT warning for web requests
os.environ["USER_AGENT"] = "MyLangChainBot/1.0 (contact: youremail@example.com)"

# ============================================================
# 2️⃣ Text Loader
# ============================================================

from langchain_community.document_loaders import TextLoader

loader = TextLoader("speech.txt")
text_documents = loader.load()
print("✅ Loaded text documents:", len(text_documents))

# ============================================================
# 3️⃣ Text Chunking
# ============================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(text_documents)
print("✅ Number of chunks:", len(documents))

# ============================================================
# 4️⃣ Embeddings + Vector Store (Chroma)
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create / Load Chroma Database
db = Chroma.from_documents(documents, embedding_model)
print("✅ Chroma vector database created.")

# ============================================================
# 5️⃣ Query the DB
# ============================================================

query = "What reason did the speaker give for entering the war?"
retrieved_results = db.similarity_search(query)
context = retrieved_results[0].page_content

print("\n🔹 Retrieved Context:\n", context[:500], "...")

# ============================================================
# 6️⃣ QA with Hugging Face Transformer (Local Model)
# ============================================================

from transformers import pipeline

# Load the QA pipeline (runs on CPU)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Ask question based on the retrieved context
qa_input = {
    "question": query,
    "context": context
}

result = qa_pipeline(qa_input)
print("\n💬 Question:", query)
print("🧠 Answer:", result["answer"])
print("✅ Confidence:", round(result["score"] * 100, 2), "%")

# ============================================================
# 7️⃣ Optional: Interactive Question Loop
# ============================================================

print("\n✨ Ask me anything about the loaded document (type 'exit' to quit)\n")

while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Retrieve most relevant chunk
    results = db.similarity_search(user_query, k=1)
    context = results[0].page_content

    # Get answer
    answer = qa_pipeline({"question": user_query, "context": context})
    print(f"Bot: {answer['answer']}  (Confidence: {round(answer['score']*100, 2)}%)\n")
