import os
import chromadb

# Initialize Persistent ChromaDB Storage
DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="blood_knowledge")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Splits long text into overlapping chunks to preserve semantic context."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_and_ingest(file_path: str):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📄 Loading document: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Generate Chunks
    chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
    print(f"🧩 Created {len(chunks)} text chunks.")

    # Prepare IDs and Metadata
    ids = [f"chunk_{i+1}" for i in range(len(chunks))]
    metadatas = [{"source": file_path, "chunk_id": i+1} for i in range(len(chunks))]

    # Store in Persistent ChromaDB
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Successfully ingested {len(chunks)} chunks into persistent DB at '{DB_PATH}'!")


if __name__ == "__main__":
    process_and_ingest("knowledge/donor_manual.txt")