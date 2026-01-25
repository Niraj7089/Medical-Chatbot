from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Step 1: Load and process PDF documents
extracted_data = load_pdf_file(data="data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# Step 2: Generate embeddings
embeddings = download_hugging_face_embeddings()  # Ensure this uses langchain_community or langchain_huggingface

# Step 3: Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create the index if it does not exist
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# Step 4: Upsert documents into Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print(f"Pinecone index '{index_name}' is ready with {len(text_chunks)} document chunks.")
