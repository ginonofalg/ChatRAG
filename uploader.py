import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import uuid
import numpy as np
import tiktoken

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "knowledge-base"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-3-small uses 1536 dimensions
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'  # replace with your preferred region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text: str, max_tokens: int = 500) -> List[str]:
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokenizer.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    
    return chunks

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def add_to_knowledge_base(text: str, metadata: dict = None):
    chunks = chunk_text(text)
    vectors = []
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        chunk_metadata = {
            "text": chunk,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        if metadata:
            chunk_metadata.update(metadata)
        vectors.append((str(uuid.uuid4()), embedding, chunk_metadata))
    
    index.upsert(vectors=vectors)

# Example usage
with open(r'C:\Users\ginon\.cursor-tutor\ChatGPTRAG\textupload3.txt', 'r') as file:
    document_text = file.read()

# Add document metadata
document_metadata = {
    "title": "Sample Document",
    "author": "John Doe",
    "date": "2024-01-20"
}

add_to_knowledge_base(document_text, metadata=document_metadata)
print(f"Document uploaded and split into chunks. Total chunks: {len(chunk_text(document_text))}")