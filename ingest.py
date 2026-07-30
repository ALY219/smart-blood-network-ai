import os
import chromadb
from pypdf import PdfReader

# Configuration
DB_PATH = "./chroma_db"
KNOWLEDGE_DIR = "knowledge"
COLLECTION_NAME = "blood_knowledge"

# Initialize Persistent ChromaDB Storage with Cosine Distance Metric
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
)


def read_file_text(file_path: str) -> str:
    """Reads raw text from either .txt or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            print(f"   ⚠️ Error reading PDF {file_path}: {e}")
            return ""
    elif ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"   ⚠️ Error reading TXT {file_path}: {e}")
            return ""
    return ""


def chunk_text(
    text: str, chunk_size: int = 300, overlap: int = 50
) -> list[str]:
    """Splits long text into overlapping chunks to preserve semantic context."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
        if chunk_size <= overlap:
            break

    return chunks


def batch_ingest_folder(folder_path: str):
    """Scans the knowledge directory and ingests all .txt and .pdf files into ChromaDB."""
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: '{folder_path}'")
        return

    supported_extensions = {".txt", ".pdf"}
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not files:
        print(f"⚠️ No .txt or .pdf files found in '{folder_path}'")
        return

    print(f"📁 Found {len(files)} document(s) in '{folder_path}' to ingest...\n")
    total_chunks_ingested = 0

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        print(f"📄 Processing: {filename}")
        raw_text = read_file_text(file_path)

        if not raw_text.strip():
            print(f"   ⚠️ Skipping {filename}: No text extracted.")
            continue

        chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
        if not chunks:
            continue

        # Prepare unique IDs and Rich Metadata
        ids = [f"{filename}_chunk_{i+1}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "file_type": ext,
                "chunk_id": i + 1,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Use collection.upsert() to update existing vectors or add new ones safely
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

        print(f"   ✅ Ingested {len(chunks)} chunk(s) with metadata.")
        total_chunks_ingested += len(chunks)

    print(
        f"\n🎉 Batch ingestion complete! Total {total_chunks_ingested} chunk(s) stored in '{DB_PATH}' (Collection: '{COLLECTION_NAME}')."
    )


if __name__ == "__main__":
    batch_ingest_folder(KNOWLEDGE_DIR)