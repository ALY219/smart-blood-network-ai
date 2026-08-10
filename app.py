import os
import re
import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import streamlit as st

# Load environment variables (.env)
load_dotenv()

# --- Page Setup ---
st.set_page_config(
    page_title="RAG Knowledge Assistant (HyDE Retrieval)",
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


@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


collection = get_chroma_collection()
ai_client = get_gemini_client()
reranker_model = get_reranker()


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


# --- HyDE Generator Utility ---
def generate_hypothetical_document(query: str, llm_client) -> str:
    """Generates an authoritative passage answering the query to align dense vector space semantics."""
    hyde_prompt = f"""Please write a detailed, authoritative paragraph from an official medical regulation or donor handbook that directly answers the following query.

Do not write meta-commentary, introductions, or disclaimers. Write directly as if it were an excerpt from the reference manual.

Query: {query}

Hypothetical Excerpt:"""

    response = llm_client.models.generate_content(
        model="gemini-2.5-flash", contents=hyde_prompt
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


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
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

    total_child_chunks = 0
    for filename in files:
        file_path = os.path.join(KNOWLEDGE_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        raw_text = read_file_text(file_path)
        if not raw_text.strip():
            continue

        parent_chunks = chunk_text(raw_text, chunk_size=600, overlap=100)

        child_documents = []
        child_ids = []
        child_metadatas = []

        for p_idx, parent_text in enumerate(parent_chunks, start=1):
            parent_id = f"{filename}_parent_{p_idx}"
            child_chunks = chunk_text(parent_text, chunk_size=150, overlap=30)

            for c_idx, child_text in enumerate(child_chunks, start=1):
                child_id = f"{parent_id}_child_{c_idx}"
                child_ids.append(child_id)
                child_documents.append(child_text)
                child_metadatas.append(
                    {
                        "source": filename,
                        "file_type": ext,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "child_idx": c_idx,
                    }
                )

        if child_documents:
            collection.upsert(
                documents=child_documents, ids=child_ids, metadatas=child_metadatas
            )
            total_child_chunks += len(child_documents)

    return total_child_chunks


# --- HyDE + Parent-Child Retrieval Engine ---
def get_hyde_context(
    query: str,
    use_hyde: bool = True,
    vector_threshold: float = 0.60,
    top_k_rerank: int = 3,
    k: int = 60,
):
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data.get("documents", [])
    all_metadatas = all_data.get("metadatas", [])
    all_ids = all_data.get("ids", [])

    if not all_docs:
        return "", ["⚠️ Vector store is empty. Ingest documents first."]

    score_logs = []

    # 1. Generate HyDE Hypothetical Document for Vector Search
    if use_hyde:
        hypothetical_doc = generate_hypothetical_document(query, ai_client)
        vector_query_text = hypothetical_doc
        score_logs.append(f"💡 **HyDE Passage:** *\"{hypothetical_doc}\"*")
    else:
        vector_query_text = query

    # 2. Dense Vector Search on Child Chunks using HyDE Text
    vector_results = collection.query(
        query_texts=[vector_query_text],
        n_results=min(15, len(all_docs)),
        include=["documents", "distances", "metadatas"],
    )

    vec_docs = vector_results["documents"][0] if vector_results["documents"] else []
    vec_ids = vector_results["ids"][0] if vector_results["ids"] else []
    vec_distances = vector_results["distances"][0] if vector_results["distances"] else []
    vec_metadatas = vector_results["metadatas"][0] if vector_results["metadatas"] else []

    vec_rank_map = {}
    child_lookup = {}
    meta_lookup = {}

    for rank, (doc_id, doc, dist, meta) in enumerate(zip(vec_ids, vec_docs, vec_distances, vec_metadatas), start=1):
        if dist <= vector_threshold:
            vec_rank_map[doc_id] = rank
            child_lookup[doc_id] = doc
            meta_lookup[doc_id] = meta

    # 3. BM25 Search on Child Chunks using Raw Query Keywords
    tokenized_corpus = [tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )

    bm25_rank_map = {}
    for rank, idx in enumerate(bm25_ranked_indices[:15], start=1):
        if bm25_scores[idx] > 0:
            doc_id = all_ids[idx]
            bm25_rank_map[doc_id] = rank
            child_lookup[doc_id] = all_docs[idx]
            meta_lookup[doc_id] = all_metadatas[idx]

    # 4. RRF Fusion on Child Matches
    all_candidate_ids = set(vec_rank_map.keys()).union(set(bm25_rank_map.keys()))
    if not all_candidate_ids:
        return "", score_logs + ["⚠️ No child chunks matched the retrieval filters."]

    rrf_scores = {}
    for doc_id in all_candidate_ids:
        score = 0.0
        if doc_id in vec_rank_map:
            score += 1.0 / (k + vec_rank_map[doc_id])
        if doc_id in bm25_rank_map:
            score += 1.0 / (k + bm25_rank_map[doc_id])
        rrf_scores[doc_id] = score

    top_child_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # 5. Map Child Candidates to Unique Parent Chunks
    unique_parents = {}
    for child_id, rrf_score in top_child_candidates:
        meta = meta_lookup[child_id]
        parent_id = meta.get("parent_id")
        parent_text = meta.get("parent_text")
        source = meta.get("source", "unknown_doc")

        if parent_id not in unique_parents:
            unique_parents[parent_id] = {
                "parent_id": parent_id,
                "parent_text": parent_text,
                "source": source,
                "max_rrf_score": rrf_score,
                "matched_child_id": child_id,
            }

    parent_list = list(unique_parents.values())

    # 6. Cross-Encoder Re-Ranking over Parent Passages
    candidate_pairs = [(query, p["parent_text"]) for p in parent_list]
    cross_encoder_scores = reranker_model.predict(candidate_pairs)

    for p_item, ce_score in zip(parent_list, cross_encoder_scores):
        p_item["ce_score"] = float(ce_score)

    final_parents = sorted(parent_list, key=lambda x: x["ce_score"], reverse=True)[:top_k_rerank]

    filtered_chunks = []
    for p_item in final_parents:
        source = p_item["source"]
        p_id = p_item["parent_id"]

        score_logs.append(
            f"🎯 **Parent Re-Rank Score:** `{p_item['ce_score']:.4f}` | Max Child RRF: `{p_item['max_rrf_score']:.5f}` | Source: `{source}` (`{p_id}`)"
        )

        formatted_chunk = f"[Source: {source} | Parent ID: {p_id}]\n{p_item['parent_text']}"
        filtered_chunks.append(formatted_chunk)

    return "\n\n---\n\n".join(filtered_chunks), score_logs


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ HyDE & Two-Stage Settings")

    enable_hyde = st.toggle(
        "Enable HyDE (Hypothetical Embeddings)",
        value=True,
        help="Generates a fake answering passage to align vector search with target documents.",
    )

    threshold = st.slider(
        "Child Vector Distance Cutoff (Cosine)",
        min_value=0.10,
        max_value=1.20,
        value=0.60,
        step=0.05,
        help="Filters weak child vector matches before mapping to Parent context.",
    )

    top_k_rerank = st.slider(
        "Final Parent Chunks for LLM",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of full Parent chunks supplied to Gemini after Cross-Encoder re-ranking.",
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
        with st.spinner("Building Parent-Child Chunks in ChromaDB..."):
            count = ingest_knowledge_folder()
            st.success(f"Ingestion Complete! {count} child chunks indexed.")

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT UI ---
st.title("🩸 Blood Donation Knowledge Assistant")
st.caption("Day 20 Architecture: HyDE Query Expansion ➔ Small Child Match ➔ Parent Context ➔ Cross-Encoder Re-Ranking")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score_logs" in msg and msg["score_logs"]:
            with st.expander("🔍 View HyDE & Retrieval Metrics"):
                for log in msg["score_logs"]:
                    st.markdown(log)

if prompt := st.chat_input("Ask a question about blood donation regulations..."):
    st.chat_message("user").markdown(prompt)

    standalone_query = contextualize_query(
        st.session_state.messages, prompt, ai_client
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        context, score_logs = get_hyde_context(
            standalone_query,
            use_hyde=enable_hyde,
            vector_threshold=threshold,
            top_k_rerank=top_k_rerank,
        )

        if standalone_query != prompt:
            score_logs.insert(
                0, f"🔄 **Standalone Query:** `{standalone_query}`"
            )

        with st.expander("🔍 View HyDE & Retrieval Metrics"):
            for log in score_logs:
                st.markdown(log)

        if not context:
            answer = "⚠️ **No context met the retrieval threshold.** I don't have enough relevant official context to answer this question."
            st.warning(answer)
        else:
            rag_prompt = f"""
Answer the user's question accurately using ONLY the context provided below.

Rules:
1. Include inline citations (e.g., [Source: donor_manual.txt | Parent ID: donor_manual.txt_parent_1]) immediately after mentioning facts from that source.
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