import chromadb
from chromadb.utils import embedding_functions

# Initialize local in-memory ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="blood_network_docs")

# Add sample medical/donor guidelines
collection.add(
    documents=[
        "O-negative blood can be donated to patients of any blood type in emergencies.",
        "Donors must wait at least 56 days between whole blood donations.",
        "AB-positive individuals are universal plasma donors."
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Semantic Query (no exact keyword overlap required)
results = collection.query(
    query_texts=["How often am I allowed to donate blood?"],
    n_results=1
)

print("Top Semantic Match:")
print(results["documents"][0][0])