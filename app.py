"""
Streamlit frontend for the Self-Healing RAG API.

Run with:
    streamlit run streamlit_app.py

Expects the FastAPI backend to be running at API_BASE_URL (default below).
"""
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Self-Healing RAG", page_icon="🔁", layout="wide")


# ----------------------------- Helpers -----------------------------

def check_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def corpus_exists(corpus_id: str) -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/corpora/{corpus_id}/exists", timeout=5)
        if r.status_code == 200:
            return r.json().get("exists", False)
    except requests.exceptions.RequestException:
        pass
    return False


def ingest_file(file, corpus_id: str):
    files = {"file": (file.name, file.getvalue())}
    data = {"corpus_id": corpus_id}
    r = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, timeout=120)
    return r


def ask_question(question: str, corpus_id: str, top_k: int, max_retries: int):
    payload = {
        "question": question,
        "corpus_id": corpus_id,
        "top_k": top_k,
        "max_retries": max_retries,
    }
    r = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=180)
    return r


def verdict_badge(verdict: str) -> str:
    return "🟢 grounded" if verdict == "grounded" else "🔴 hallucinated"


# ----------------------------- Session state -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of AskResponse dicts

if "corpus_id" not in st.session_state:
    st.session_state.corpus_id = "default_corpus"


# ----------------------------- Sidebar -----------------------------

with st.sidebar:
    st.title("🔁 Self-Healing RAG")
    st.caption("Retrieve → Generate → Critique → Retry")

    healthy = check_health()
    if healthy:
        st.success(f"Backend reachable\n\n`{API_BASE_URL}`")
    else:
        st.error(f"Cannot reach backend at `{API_BASE_URL}`.\nIs `uvicorn app.main:app` running?")

    st.divider()

    st.subheader("1. Corpus")
    st.session_state.corpus_id = st.text_input(
        "Corpus ID", value=st.session_state.corpus_id, help="A label grouping ingested documents together."
    )

    if st.session_state.corpus_id:
        if corpus_exists(st.session_state.corpus_id):
            st.caption("✅ This corpus has documents.")
        else:
            st.caption("⚠️ This corpus is empty. Ingest a document below.")

    st.subheader("2. Ingest a document")
    uploaded_file = st.file_uploader("Upload .txt, .md, or .pdf", type=["txt", "md", "pdf"])

    if st.button("Ingest document", disabled=(uploaded_file is None or not healthy), use_container_width=True):
        with st.spinner(f"Chunking and embedding '{uploaded_file.name}'..."):
            try:
                resp = ingest_file(uploaded_file, st.session_state.corpus_id)
                if resp.status_code == 200:
                    body = resp.json()
                    st.success(f"Ingested {body['num_chunks']} chunks into '{body['corpus_id']}'.")
                else:
                    st.error(f"Ingestion failed ({resp.status_code}): {resp.json().get('detail', resp.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    st.divider()

    st.subheader("3. Retrieval / retry settings")
    top_k = st.slider("top_k (chunks retrieved)", min_value=1, max_value=10, value=4)
    max_retries = st.slider("max_retries", min_value=0, max_value=5, value=2)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ----------------------------- Main area -----------------------------

st.header("Ask a question")

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Your question", placeholder="e.g. What does the document say about refund policy?")
    submitted = st.form_submit_button("Ask", use_container_width=True, disabled=not healthy)

if submitted and question.strip():
    with st.spinner("Retrieving, generating, and critiquing the answer..."):
        try:
            resp = ask_question(question.strip(), st.session_state.corpus_id, top_k, max_retries)
            if resp.status_code == 200:
                st.session_state.history.insert(0, resp.json())
            elif resp.status_code == 404:
                st.error(f"Corpus '{st.session_state.corpus_id}' is empty. Ingest a document first.")
            else:
                st.error(f"Request failed ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

st.divider()

if not st.session_state.history:
    st.info("No questions asked yet. Ingest a document in the sidebar, then ask a question above.")

for item in st.session_state.history:
    with st.container(border=True):
        st.markdown(f"**Q:** {item['question']}")

        if item["was_fallback"]:
            st.warning(f"**A:** {item['final_answer']}")
        else:
            st.success(f"**A:** {item['final_answer']}")

        col1, col2 = st.columns(2)
        col1.caption(f"Total attempts: {item['total_attempts']}")
        col2.caption(f"Fell back to 'insufficient info': {'Yes' if item['was_fallback'] else 'No'}")

        with st.expander(f"🔍 See self-healing trace ({len(item['attempts'])} attempt(s))"):
            for attempt in item["attempts"]:
                crit = attempt["critique"]
                st.markdown(
                    f"**Attempt {attempt['attempt_number']}** — "
                    f"{'✅ accepted' if attempt['accepted'] else '❌ rejected'} — "
                    f"{verdict_badge(crit['verdict'])}"
                )
                st.text(f"Query used: {attempt['query_used']}")

                m1, m2, m3 = st.columns(3)
                m1.metric("LLM faithfulness", f"{crit['llm_faithfulness_score']:.2f}")
                m2.metric("Embedding grounding", f"{crit['embedding_grounding_score']:.2f}")
                m3.metric("Combined score", f"{crit['combined_score']:.2f}")

                if crit["reasoning"]:
                    st.caption(f"Critic reasoning: {crit['reasoning']}")

                if crit["unsupported_claims"]:
                    st.markdown("**Unsupported claims flagged:**")
                    for claim in crit["unsupported_claims"]:
                        st.markdown(f"- {claim}")

                st.markdown(f"**Answer produced this attempt:** {attempt['answer']}")

                st.markdown(f"**Retrieved chunks ({len(attempt['retrieved_chunks'])}):**")
                for chunk in attempt["retrieved_chunks"]:
                    st.markdown(
                        f"> *source: {chunk['source']} | distance: {chunk['distance']:.3f}*\n>\n> {chunk['text'][:400]}"
                        + ("..." if len(chunk["text"]) > 400 else "")
                    )

                st.markdown("---")