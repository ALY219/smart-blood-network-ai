import chromadb

# Initialize ChromaDB in-memory client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="blood_knowledge")

def seed_knowledge_base():
    """Seeds local vector database with blood donation guidelines if empty."""
    if collection.count() == 0:
        collection.add(
            documents=[
                "O-negative is the universal red blood cell donor type and can be given to patients of any blood type in emergency situations.",
                "Donors must wait at least 56 days (8 weeks) between whole blood donations to allow iron levels to recover.",
                "AB-positive individuals are universal plasma donors and can receive red blood cells from any ABO blood group.",
                "Standard donation eligibility requires donors to be aged 18 to 65, weigh at least 50 kg, and have a hemoglobin level above 12.5 g/dL."
            ],
            ids=["policy_01", "policy_02", "policy_03", "policy_04"]
        )
        print("✅ RAG Knowledge Base seeded successfully!")

def get_relevant_context(user_query: str, n_results: int = 2) -> str:
    """Retrieves top matching document snippets for a user prompt."""
    results = collection.query(query_texts=[user_query], n_results=n_results)
    docs = results.get("documents", [[]])[0]
    if not docs:
        return "No specific policy guidelines found."
    return "\n".join(f"- {doc}" for doc in docs)