import os
import chromadb
from dotenv import load_dotenv
from google import genai

# 1. Load Environment Variables (.env file)
load_dotenv()

# 2. Configuration
DB_PATH = "./chroma_db"
COLLECTION_NAME = "blood_knowledge"

# 3. Initialize Clients
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Retrieve the collection configured with Cosine distance metric
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)

# Gemini Client (automatically uses GEMINI_API_KEY from environment/.env)
ai_client = genai.Client()


# 4. Distance Thresholding & Citation Header Formatting
def get_relevant_context(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
    threshold: float = 0.5,  # Cosine Distance: 0.0 = exact match, <= 0.5 = strong semantic match
) -> str:
    """Queries ChromaDB, filters chunks by distance threshold, and formats source metadata."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    filtered_chunks = []

    print(f"\n🔍 Query: '{query}'")
    print(f"📏 Filtering with Cosine distance threshold <= {threshold}\n")

    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
        source = meta.get("source", "unknown_doc")
        chunk_id = meta.get("chunk_id", meta.get("chunk", i + 1))

        if dist <= threshold:
            # Prepend source header for Gemini citation
            formatted_chunk = f"[Source: {source} | Chunk {chunk_id}]\n{doc}"
            filtered_chunks.append(formatted_chunk)
            print(
                f"  ✅ [INJECTED] Distance: {dist:.4f} | {source} (Chunk {chunk_id})"
            )
        else:
            print(
                f"  ❌ [IGNORED]  Distance: {dist:.4f} | {source} (Chunk {chunk_id})"
            )

    if not filtered_chunks:
        print("\n⚠️ No context met the threshold.")
        return ""

    return "\n\n---\n\n".join(filtered_chunks)


# 5. RAG Pipeline Generation
def answer_query(query: str, threshold: float = 0.5):
    """Retrieves relevant context and generates a grounded response with Gemini."""
    context = get_relevant_context(query, collection, threshold=threshold)

    # Block prompt execution if no context meets the threshold requirements
    if not context:
        print(
            "\n🤖 Gemini Answer: I don't have enough relevant official context to answer this question."
        )
        return

    prompt = f"""
Answer the user's question accurately using ONLY the context provided below.

Rules:
1. Include inline citations (e.g., [Source: donor_manual.txt | Chunk 2]) immediately after mentioning facts from that source.
2. Do not use outside knowledge. If the context does not contain the answer, state that the information is unavailable in the provided documents.

Context:
{context}

User Question: {query}
"""

    response = ai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print("\n🤖 Gemini Answer:")
    print(response.text)


if __name__ == "__main__":
    # Test Question
    test_query = "What are the eligibility requirements for donors?"
    answer_query(test_query, threshold=1.05)