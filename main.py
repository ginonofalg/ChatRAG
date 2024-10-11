import os
from typing import List
import openai
from dotenv import load_dotenv
import pinecone
import tiktoken
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index object
index_name = "knowledge-base"
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
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # Changed from response['data'][0]['embedding']

def retrieve_relevant_context(query: str, k: int = 3) -> List[str]:
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [match['metadata']['text'] for match in results['matches']]

def generate_response(conversation_history, relevant_context):
    messages = conversation_history + [{"role": "system", "content": "Relevant context: " + ' '.join(relevant_context)}]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    #return response.choices[0].message.content.strip()
    return response.choices[0].message.content.strip()

def chatbot():
    print("Chatbot: Hello! How can I assist you today? (Type 'quit' to exit)")
    
    conversation_history = []
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        
        conversation_history.append({"role": "user", "content": user_input})
        
        relevant_context = retrieve_relevant_context(user_input)
        response = generate_response(conversation_history, relevant_context)
        
        print(f"Chatbot: {response}")
        
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    chatbot()
