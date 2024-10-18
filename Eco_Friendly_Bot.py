import pinecone
import os
import openai
import streamlit as st
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone using the Pinecone class and ServerlessSpec
pc = Pinecone(api_key="73768607-1a57-40e9-9f5a-35892673a361")
openai.api_key = "sk-uvd31fcl6TA3ANFk7f1PU8_Ej-fKLfCZivT9XWJEUST3BlbkFJH8CMSJBKFIqk6XJOW__WwQ425AYiOR-VyywYJBCBcA"
# Define your index name, ensure it follows the naming convention
index_name = 'my-index'  # Must be lowercase, alphanumeric, or contain '-'

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Define the dimension of your vectors
        metric='euclidean',  # Similarity metric
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Define the cloud region
    )

# Connect to the index
index = pc.Index(index_name)

# Example vector data for upserting (ensure dimension matches)
vector_data = [
    {"id": "vec1", "values": [float(i) for i in range(1536)]},  # Vector dimension must match the index's dimension
    {"id": "vec2", "values": [float(i) for i in range(1536,3072)]}
]

# Upsert vectors to the index
index.upsert(vectors=vector_data)

# Query the index (using the first vector as an example)
query_vector = [float(i) for i in range(1536)]
query_result = index.query(vector=query_vector, top_k=3)
print(query_result)


# Now you can use `index` to interact with Pinecone, e.g., upserting vectors or querying


# Function to generate embeddings
def generate_embeddings(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Get user input
user_input = input("Please enter your query: ")

# Generate embeddings for the input
embedding = generate_embeddings(user_input)

# Upsert or query the index as needed
index.upsert([(user_input, embedding)])

# Function to query Pinecone and retrieve tips
def query_pinecone(index, user_query):
    # Generate embeddings using Groq API
    embeddings = generate_embeddings(user_query)
    
    # Perform Pinecone query, return top 3 results
    response = index.query(embeds=[embeddings], top_k=3)
    return response

# Display tips based on user input
if user_input:
    # Query Pinecone for relevant tips
    response = query_pinecone(index, user_input)
    
    # Display the matching tips
    st.write("Here are some eco-friendly tips for you:")
    if response and 'matches' in response:
        for match in response['matches']:
            st.write(f"Tip: {match['id']} (Score: {match['score']})")
    else:
        st.write("No relevant tips found. Please try again.")
