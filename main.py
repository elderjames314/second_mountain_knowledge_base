import streamlit as st
import openai
import pandas as pd
import faiss
import numpy as np
from docx import Document
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the correct key from .env
OPENAI_API_KEY = os.getenv('OPEN_API_KEY')

# Load your OpenAI API key
openai.api_key = OPENAI_API_KEY



@st.cache_data
def load_documents():
    with open("docs/trackingcapitalflow.txt", "r") as file:
        tracking_data = file.read()

    # Load docx files
    def load_docx(filepath):
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    capitalflows_data = load_docx("docs/capitalflows.docx")
    pastcycles_data = load_docx("docs/pastcyclestudies.docx")

    return [tracking_data, capitalflows_data, pastcycles_data]

# Generate embeddings using OpenAI's API


def generate_embeddings(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Create FAISS index for the text sections


@st.cache_data
def create_embeddings_index(texts):
    embeddings = []
    for text in texts:
        embeddings.append(generate_embeddings(text))

    embeddings_np = np.array(embeddings).astype("float32")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    return index, embeddings_np

# Search function to find the most relevant section


def search_query(query, index, texts, embeddings, model="text-embedding-ada-002", top_k=5):
    # Generate the query embedding
    query_embedding = generate_embeddings(query, model=model)
    query_embedding_np = np.array([query_embedding]).astype("float32")

    # Search for the top k most relevant sections
    distances, indices = index.search(query_embedding_np, k=top_k)

    # Retrieve the top k sections and their distances
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append((texts[idx], distance))

    # Sort results by similarity (distance is lower for more similar results)
    results.sort(key=lambda x: x[1])  # Sort by distance in ascending order

    return results

# Streamlit UI


# Streamlit UI
def main():
    st.title("Second Mountain Knowledge Base")
    st.write("This tool allows you to query information about second mountain across blockchains and past cycles.")

    # Load document content
    documents = load_documents()
    sections = []
    for doc in documents:
        sections.extend(doc.split("\n\n"))  # Split documents into sections

    # Create FAISS index
    index, embeddings = create_embeddings_index(sections)

    # User query input
    query = st.text_input("Enter your query here",
                          "How does capital flow across blockchain networks?")

    # Search and display the results
    if st.button("Search"):
        if query:
            top_k = 5  # Number of top results to display
            results = search_query(
                query, index, sections, embeddings, top_k=top_k)

            st.subheader(f"Top {top_k} Matching Sections:")
            for i, (result, distance) in enumerate(results, start=1):
                st.write(f"**Result {i}:**")
                st.write(result)
                # Convert distance to similarity (1 - distance)
                st.write(f"Similarity Score (distance): {1 - distance:.4f}")
                st.write("---")


if __name__ == "__main__":
    main()
