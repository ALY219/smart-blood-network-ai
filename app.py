import os
import re
import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
import streamlit as st

# Load environment variables (.env)
load_dotenv()

# --- Page Setup ---
st.set_page_config(
    page_title="RAG Knowledge Assistant (Hybrid Search)",
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


# --- BM25 Tokenizer Utility ---
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# --- Contextual Query Rewriter ---
def contextualize_query(chat_history: list, latest_question: str, llm_client) -> str:
    if not chat_history:
        return latest_question

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


# --- Hybrid Retrieval Engine (BM25 + Chroma Vector + RRF) ---
def get_hybrid_context(query: str, vector_threshold: float = 0.60, k: int = 60):
    # 1. Fetch All Documents from ChromaDB to build in-memory BM25 index
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data.get("documents", [])
    all_metadatas = all_data.get("metadatas", [])
    all_ids = all_data.get("ids", [])

    if not all_docs:
        return "", ["⚠️ Vector store is empty. Ingest documents first."]

    # 2. Dense Vector Search (ChromaDB)
    vector_results = collection.query(
        query_texts=[query],
        n_results=min(10, len(all_docs)),
        include=["documents", "distances", "metadatas"],
    )

    vec_docs = vector_results["documents"][0] if vector_results["documents"] else []
    vec_ids = vector_results["ids"][0] if vector_results["ids"] else []
    vec_distances = vector_results["distances"][0] if vector_results["distances"] else []
    vec_metadatas = vector_results["metadatas"][0] if vector_results["metadatas"] else []

    # Map Vector IDs to Ranks and Metadata
    vec_rank_map = {}
    doc_lookup = {}
    meta_lookup = {}

    for rank, (doc_id, doc, dist, meta) in enumerate(zip(vec_ids, vec_docs, vec_distances, vec_metadatas), start=1):
        if dist <= vector_threshold:
            vec_rank_map[doc_id] = rank
            doc_lookup[doc_id] = doc
            meta_lookup[doc_id] = meta

    # 3. Sparse Keyword Search (BM25)
    tokenized_corpus = [tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # Sort documents by BM25 score descending
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )

    bm25_rank_map = {}
    for rank, idx in enumerate(bm25_ranked_indices[:10], start=1):
        if bm25_scores[idx] > 0:  # Only include non-zero keyword matches
            doc_id = all_ids[idx]
            bm25_rank_map[doc_id] = rank
            doc_lookup[doc_id] = all_docs[idx]
            meta_lookup[doc_id] = all_metadatas[idx]

    # 4. Reciprocal Rank Fusion (RRF)
    all_candidate_ids = set(vec_rank_map.keys()).union(set(bm25_rank_map.keys()))
    rrf_scores = {}

    for doc_id in all_candidate_ids:
        score = 0.0
        if doc_id in vec_rank_map:
            score += 1.0 / (k + vec_rank_map[doc_id])
        if doc_id in bm25_rank_map:
            score += 1.0 / (k + bm25_rank_map[doc_id])
        rrf_scores[doc_id] = score

    # Sort by highest RRF score
    sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    filtered_chunks = []
    score_logs = []

    for doc_id, rrf_score in sorted_candidates[:5]:
        doc = doc_lookup[doc_id]
        meta = meta_lookup[doc_id]
        source = meta.get("source", "unknown_doc")
        chunk_id = meta.get("chunk_id", 1)

        vec_rank_str = f"Rank {vec_rank_map[doc_id]}" if doc_id in vec_rank_map else "N/A"
        bm25_rank_str = f"Rank {bm25_rank_map[doc_id]}" if doc_id in bm25_rank_map else "N/A"

        score_logs.append(
            f"⚡ **RRF Score:** `{rrf_score:.5f}` | **Vector:** `{vec_rank_str}` | **BM25:** `{bm25_rank_str}` | Source: `{source}` (Chunk {chunk_id})"
        )

        formatted_chunk = f"[Source: {source} | Chunk {chunk_id}]\n{doc}"
        filtered_chunks.append(formatted_chunk)

    return "\n\n---\n\n".join(filtered_chunks), score_logs


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ Hybrid RAG Settings")

    threshold = st.slider(
        "Vector Distance Cutoff (Cosine)",
        min_value=0.10,
        max_value=1.20,
        value=0.60,
        step=0.05,
        help="Filters out weak semantic vector matches before RRF fusion.",
    )

    st.divider()

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
st.caption("Hybrid Search Engine: BM25 Keyword Search + ChromaDB Vector Search via Reciprocal Rank Fusion (RRF)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score_logs" in msg and msg["score_logs"]:
            with st.expander("🔍 View Hybrid Search & RRF Metrics"):
                for log in msg["score_logs"]:
                    st.markdown(log)

if prompt := st.chat_input("Ask a question about blood donation regulations..."):
    st.chat_message("user").markdown(prompt)

    standalone_query = contextualize_query(
        st.session_state.messages, prompt, ai_client
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        context, score_logs = get_hybrid_context(
            standalone_query, vector_threshold=threshold
        )

        if standalone_query != prompt:
            score_logs.insert(
                0, f"🔄 **Standalone Query:** `{standalone_query}`"
            )

        with st.expander("🔍 View Hybrid Search & RRF Metrics"):
            for log in score_logs:
                st.markdown(log)

        if not context:
            answer = "⚠️ **No context met the hybrid threshold.** I don't have enough relevant official context to answer this question."
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