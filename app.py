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
    page_title="RAG Knowledge Assistant (Corrective RAG)",
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


# --- Multi-Query Generator Utility ---
def generate_multi_queries(query: str, llm_client, num_queries: int = 3) -> list[str]:
    multi_prompt = f"""You are an AI assistant helping optimize search queries for a document retrieval engine.
Generate {num_queries} different versions or alternative perspectives of the user query below to retrieve relevant documents from a knowledge base.

Provide each variation on a new line. Do not number them or add preambles.

Original Query: {query}

Query Variations:"""

    response = llm_client.models.generate_content(
        model="gemini-2.5-flash", contents=multi_prompt
    )

    variations = [q.strip("- ").strip() for q in response.text.strip().split("\n") if q.strip()]

    all_queries = [query]
    for v in variations:
        if v.lower() != query.lower() and v not in all_queries:
            all_queries.append(v)

    return all_queries[: num_queries + 1]


# --- HyDE Generator Utility ---
def generate_hypothetical_document(query: str, llm_client) -> str:
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


# --- Corrective RAG (CRAG) Engine ---
def get_crag_context(
    query: str,
    use_multi_query: bool = True,
    use_hyde: bool = True,
    vector_threshold: float = 0.60,
    crag_high_threshold: float = 0.0,
    crag_low_threshold: float = -2.5,
    top_k_rerank: int = 3,
    k: int = 60,
):
    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data.get("documents", [])
    all_metadatas = all_data.get("metadatas", [])
    all_ids = all_data.get("ids", [])

    if not all_docs:
        return "", "INCORRECT", ["⚠️ Vector store is empty. Ingest documents first."]

    score_logs = []

    # 1. Multi-Query Expansion
    if use_multi_query:
        queries = generate_multi_queries(query, ai_client, num_queries=3)
        score_logs.append("🔀 **Multi-Query Variations:**")
        for idx, q_var in enumerate(queries, start=1):
            score_logs.append(f"  {idx}. `{q_var}`")
    else:
        queries = [query]

    tokenized_corpus = [tokenize(doc) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    child_rrf_accumulator = {}
    child_lookup = {}
    meta_lookup = {}

    # 2. Multi-Stream Retrieval
    for sub_query in queries:
        search_vector_text = (
            generate_hypothetical_document(sub_query, ai_client)
            if use_hyde
            else sub_query
        )

        vector_results = collection.query(
            query_texts=[search_vector_text],
            n_results=min(10, len(all_docs)),
            include=["documents", "distances", "metadatas"],
        )

        vec_docs = vector_results["documents"][0] if vector_results["documents"] else []
        vec_ids = vector_results["ids"][0] if vector_results["ids"] else []
        vec_distances = vector_results["distances"][0] if vector_results["distances"] else []
        vec_metadatas = vector_results["metadatas"][0] if vector_results["metadatas"] else []

        for rank, (doc_id, doc, dist, meta) in enumerate(
            zip(vec_ids, vec_docs, vec_distances, vec_metadatas), start=1
        ):
            if dist <= vector_threshold:
                child_lookup[doc_id] = doc
                meta_lookup[doc_id] = meta
                child_rrf_accumulator[doc_id] = child_rrf_accumulator.get(doc_id, 0.0) + (1.0 / (k + rank))

        tokenized_sub_query = tokenize(sub_query)
        bm25_scores = bm25.get_scores(tokenized_sub_query)
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )

        for rank, idx in enumerate(bm25_ranked_indices[:10], start=1):
            if bm25_scores[idx] > 0:
                doc_id = all_ids[idx]
                child_lookup[doc_id] = all_docs[idx]
                meta_lookup[doc_id] = all_metadatas[idx]
                child_rrf_accumulator[doc_id] = child_rrf_accumulator.get(doc_id, 0.0) + (1.0 / (k + rank))

    if not child_rrf_accumulator:
        return "", "INCORRECT", score_logs + ["⚠️ No child chunks passed distance thresholds."]

    top_child_candidates = sorted(
        child_rrf_accumulator.items(), key=lambda x: x[1], reverse=True
    )[:12]

    # 3. Map to Parent Chunks
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
            }

    parent_list = list(unique_parents.values())

    # 4. Cross-Encoder Evaluation & Scoring
    candidate_pairs = [(query, p["parent_text"]) for p in parent_list]
    cross_encoder_scores = reranker_model.predict(candidate_pairs)

    for p_item, ce_score in zip(parent_list, cross_encoder_scores):
        p_item["ce_score"] = float(ce_score)

    final_parents = sorted(parent_list, key=lambda x: x["ce_score"], reverse=True)[:top_k_rerank]
    top_score = final_parents[0]["ce_score"] if final_parents else -999.0

    # 5. CRAG Decision Routing
    if top_score >= crag_high_threshold:
        crag_status = "CORRECT"
    elif top_score >= crag_low_threshold:
        crag_status = "AMBIGUOUS"
    else:
        crag_status = "INCORRECT"

    score_logs.append(f"🛡️ **CRAG Assessment:** `{crag_status}` (Top Cross-Encoder Score: `{top_score:.4f}`)")

    if crag_status == "INCORRECT":
        return "", crag_status, score_logs

    filtered_chunks = []
    for p_item in final_parents:
        source = p_item["source"]
        p_id = p_item["parent_id"]

        score_logs.append(
            f"🎯 **Parent Re-Rank Score:** `{p_item['ce_score']:.4f}` | Source: `{source}` (`{p_id}`)"
        )
        formatted_chunk = f"[Source: {source} | Parent ID: {p_id}]\n{p_item['parent_text']}"
        filtered_chunks.append(formatted_chunk)

    return "\n\n---\n\n".join(filtered_chunks), crag_status, score_logs


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ CRAG Architecture Settings")

    enable_multi_query = st.toggle("Enable Multi-Query Expansion", value=True)
    enable_hyde = st.toggle("Enable HyDE Transformation", value=True)

    st.subheader("🛡️ CRAG Confidence Thresholds")
    crag_high = st.slider(
        "High Confidence Threshold (CORRECT)",
        min_value=-2.0,
        max_value=3.0,
        value=0.0,
        step=0.25,
        help="Scores above this tier trigger full grounded answer generation.",
    )

    crag_low = st.slider(
        "Low Confidence Cutoff (INCORRECT)",
        min_value=-5.0,
        max_value=0.0,
        value=-2.5,
        step=0.25,
        help="Scores below this tier drop context to prevent hallucination.",
    )

    threshold = st.slider(
        "Child Vector Distance Cutoff (Cosine)",
        min_value=0.10,
        max_value=1.20,
        value=0.60,
        step=0.05,
    )

    top_k_rerank = st.slider(
        "Final Parent Chunks for LLM", min_value=1, max_value=5, value=3, step=1
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
st.caption("Day 22 Architecture: Corrective RAG (CRAG) Guidance + Multi-Query + HyDE + Parent-Child Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "score_logs" in msg and msg["score_logs"]:
            with st.expander("🔍 View CRAG Metrics & Evaluation Logs"):
                for log in msg["score_logs"]:
                    st.markdown(log)

if prompt := st.chat_input("Ask a question about blood donation regulations..."):
    st.chat_message("user").markdown(prompt)

    standalone_query = contextualize_query(
        st.session_state.messages, prompt, ai_client
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        context, crag_status, score_logs = get_crag_context(
            standalone_query,
            use_multi_query=enable_multi_query,
            use_hyde=enable_hyde,
            vector_threshold=threshold,
            crag_high_threshold=crag_high,
            crag_low_threshold=crag_low,
            top_k_rerank=top_k_rerank,
        )

        if standalone_query != prompt:
            score_logs.insert(0, f"🔄 **Standalone Query:** `{standalone_query}`")

        with st.expander("🔍 View CRAG Metrics & Evaluation Logs"):
            for log in score_logs:
                st.markdown(log)

        # Corrective Action Execution
        if crag_status == "INCORRECT":
            answer = "⚠️ **Corrective RAG Warning:** The retrieved context failed quality relevance thresholds. To avoid hallucination, I cannot provide an official answer based on the current documents."
            st.error(answer)

        elif crag_status == "AMBIGUOUS":
            st.warning("⚠️ **Corrective RAG Notice:** Retrieval confidence is ambiguous. Answer generated with caution constraints.")
            rag_prompt = f"""
Answer the user's question using ONLY the provided context below.

CRAG NOTICE: The provided context may only partially address the question.
1. State clearly what is supported directly by the context.
2. Explicitly highlight any missing or uncertain details that are not fully confirmed in the context.
3. Include inline citations (e.g., [Source: donor_manual.txt | Parent ID: donor_manual.txt_parent_1]).

Context:
{context}

User Question: {prompt}
"""
            with st.spinner("Thinking cautiously..."):
                response = ai_client.models.generate_content(
                    model="gemini-2.5-flash", contents=rag_prompt
                )
                answer = response.text
            st.markdown(answer)

        else:  # CORRECT
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