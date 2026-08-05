import os
import chromadb
from dotenv import load_dotenv
from google import genai
import streamlit as st

# Load environment variables (.env)
load_dotenv()

# --- Page Setup ---
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🩸",
    layout="wide",
)

DB_PATH = "./chroma_db"
KNOWLEDGE_DIR = "knowledge"
COLLECTION_NAME = "blood_knowledge"

os.makedirs(KNOWLEDGE_DIR, exist_ok=True)


# --- Cached Resource Initialization ---
@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


@st.cache_resource
def get_gemini_client():
    return genai.Client()


collection = get_chroma_collection()
ai_client = get_gemini_client()


# --- Contextual Query Rewriter (Day 16) ---
def contextualize_query(chat_history: list, latest_question: str, llm_client) -> str:
    """Transforms follow-up questions into standalone search queries using past chat history."""
    if not chat_history:
        return latest_question

    # Extract up to the last 2 conversation turns (4 messages total)
    recent_history = [
        msg for msg in chat_history if msg["role"] in ["user", "assistant"]
    ][-4:]

    if not recent_history:
        return latest_question

    formatted_history = ""
    for msg in recent_history:
        formatted_history += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt = f"""Given the following chat history and a follow-up question, rephrase the follow-up question to be a STANDALONE search query that contains all necessary context for document retrieval.

Do NOT answer the question, only rephrase it into a concise standalone search query.

Chat History:
{formatted_history}

Follow-Up Question: {latest_question}

Standalone Query:"""

    response = llm_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    return response.text.strip()


# --- Ingestion Utilities ---
def read_file_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def chunk_text(
    text: str, chunk_size: int = 300, overlap: int = 50
) -> list[str]:
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


def ingest_knowledge_folder():
    files = [
        f
        for f in os.listdir(KNOWLEDGE_DIR)
        if os.path.splitext(f)[1].lower() in {".txt", ".pdf"}
    ]
    if not files:
        return 0

    total_chunks = 0
    for filename in files:
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        raw_text = read_file_text(file_path)
        if not raw_text.strip():
            continue

        chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
        if not chunks:
            continue

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

        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)

    return total_chunks


# --- Retrieval Utility ---
def get_relevant_context(query: str, threshold: float = 0.60):
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    filtered_chunks = []
    score_logs = []

    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metadatas)):
        source = meta.get("source", "unknown_doc")
        chunk_id = meta.get("chunk_id", meta.get("chunk", i + 1))
        passed = dist <= threshold

        status = "✅ INJECTED" if passed else "❌ IGNORED"
        score_logs.append(
            f"**{status}** | Distance: `{dist:.4f}` | Source: `{source}` (Chunk {chunk_id})"
        )

        if passed:
            formatted_chunk = f"[Source: {source} | Chunk {chunk_id}]\n{doc}"
            filtered_chunks.append(formatted_chunk)

    return "\n\n---\n\n".join(filtered_chunks), score_logs


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ RAG Settings")

    # Interactive Threshold Slider
    threshold = st.slider(
        "Distance Threshold (Cosine)",
        min_value=0.10,
        max_value=1.20,
        value=0.60,
        step=0.05,
        help="Scores lower than or equal to this threshold are injected into Gemini's context.",
    )

    st.divider()

    # Upload & Ingest UI
    st.subheader("📄 Upload Knowledge Files")
    uploaded_files = st.file_uploader(
        "Add .pdf or .txt files", type=["pdf", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        for uf in uploaded_files:
            file_path = os.path.join(KNOWLEDGE_DIR, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to 'knowledge/'!")

    if st.button("🚀 Re-Index Knowledge Base", use_container_width=True):
        with st.spinner("Processing documents into ChromaDB..."):
            count = ingest_knowledge_folder()
            st.success(f"Ingestion Complete! {count} total chunks stored.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---
st.title("🩸 Blood Donation Knowledge Assistant")
st.caption("Grounded QA with Multi-Turn Conversational Memory & ChromaDB Vector Search")

# Chat Session History Init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score_logs" in msg and msg["score_logs"]:
            with st.expander("🔍 View Search Scores & Distance Metrics"):
                for log in msg["score_logs"]:
                    st.markdown(log)

# Input Box
if prompt := st.chat_input("Ask a question about blood donation regulations..."):
    # 1. Render user prompt immediately
    st.chat_message("user").markdown(prompt)

    # 2. Rephrase follow-up query using historical context BEFORE adding prompt to session
    standalone_query = contextualize_query(
        st.session_state.messages, prompt, ai_client
    )

    # Append original user prompt to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        context, score_logs = get_relevant_context(
            standalone_query, threshold=threshold
        )

        # Prepend query rewriting metadata log if modified
        if standalone_query != prompt:
            score_logs.insert(
                0, f"🔄 **Standalone Query:** `{standalone_query}`"
            )

        with st.expander("🔍 View Search Scores & Distance Metrics"):
            for log in score_logs:
                st.markdown(log)

        if not context:
            answer = "⚠️ **No context met the distance threshold.** I don't have enough relevant official context to answer this question."
            st.warning(answer)
        else:
            rag_prompt = f"""
Answer the user's question accurately using ONLY the context provided below.

Rules:
1. Include inline citations (e.g., [Source: donor_manual.txt | Chunk 2]) immediately after mentioning facts from that source.
2. Do not use outside knowledge. If the context does not contain the answer, state that the information is unavailable in the provided documents.

Context:
{context}

User Question: {prompt}
"""
            with st.spinner("Thinking..."):
                response = ai_client.models.generate_content(
                    model="gemini-2.5-flash", contents=rag_prompt
                )
                answer = response.text

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "score_logs": score_logs}
        )