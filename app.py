import logging
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents.evaluator import Evaluator
from agents.generator import Generator
from agents.memory import EvaluatorMemory, GeneratorMemory
from ingestion.loaders import SUPPORTED_EXTENSIONS, load_file
from ingestion.web import load_web_page, load_wikipedia_page
from knowledge import cache
from knowledge.vector_store import add_documents, count_chunks, list_sources
from workflow.graph import MAX_ITERATIONS, WorkflowError, run_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("app")

st.set_page_config(page_title="Evaluator-Generator Assistant", layout="wide")

if not os.getenv("OPENROUTER_API_KEY", "").strip():
    st.error("Please set the OPENROUTER_API_KEY environment variable before starting the app.")
    st.stop()

# every browser session gets its own agents, so the memories are not shared
if "generator" not in st.session_state:
    st.session_state.generator = Generator(GeneratorMemory())
    st.session_state.evaluator = Evaluator(EvaluatorMemory())
    st.session_state.last_result = None
    st.session_state.metrics = {
        "questions": 0, "cache_hits": 0, "approved": 0,
        "unvalidated": 0, "iterations": 0, "seconds": 0.0,
    }

generator = st.session_state.generator
evaluator = st.session_state.evaluator
metrics = st.session_state.metrics


# adding knowledge
def ingest(documents, label):
    stats = add_documents(documents)
    if stats["chunks"] == 0:
        st.warning(f"{label}: no text could be extracted.")
    elif stats["new"] == 0:
        st.info(f"{label}: already in the knowledge base ({stats['existing']} chunks), nothing added.")
    else:
        message = f"{label}: added {stats['new']} new chunks"
        if stats["existing"]:
            message += f" ({stats['existing']} already existed)."
        else:
            message += "."
        st.success(message)


def process_uploaded_file(uploaded_file):
    if uploaded_file.size > 50 * 1024 * 1024:
        raise ValueError(f"The file is larger than {50} MB.")

    # the loaders need a real file on the disk, so we save a temporary copy
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)
    try:
        documents = load_file(temp_path, display_name=uploaded_file.name)
        ingest(documents, uploaded_file.name)
    finally:
        temp_path.unlink(missing_ok=True)


def files_tab():
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT, PowerPoint, source code, or WAV audio",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if not st.button("Process uploaded files"):
        return
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return
    with st.spinner("Processing files (audio can take a while)..."):
        for uploaded_file in uploaded_files:
            try:
                process_uploaded_file(uploaded_file)
            except Exception as error:
                logger.exception("Could not process %s", uploaded_file.name)
                st.error(f"Could not process {uploaded_file.name}: {error}")


def web_tab():
    url = st.text_input("Web page URL", placeholder="https://example.com/article")
    if not st.button("Add web page"):
        return
    if not url.strip():
        st.warning("Please enter a URL.")
        return
    try:
        with st.spinner("Reading web page..."):
            ingest(load_web_page(url), url.strip())
    except Exception as error:
        logger.exception("Could not load web page")
        st.error(f"Could not load the web page: {error}")


def wikipedia_tab():
    title = st.text_input("Wikipedia page title or URL", placeholder="Artificial intelligence")
    language = st.selectbox("Language (used when you type a title)", ["en", "ar", "fr", "de", "es"])
    if not st.button("Add Wikipedia page"):
        return
    if not title.strip():
        st.warning("Please enter a page title or URL.")
        return
    try:
        with st.spinner("Reading Wikipedia..."):
            ingest(load_wikipedia_page(title, language), title.strip())
    except Exception as error:
        logger.exception("Could not load Wikipedia page")
        st.error(f"Could not load the Wikipedia page: {error}")


# the sidebar (status and monitoring)
def sidebar():
    st.header("System status")

    if cache.is_available():
        st.success("Redis cache: connected")
    else:
        st.warning("Redis cache: not reachable. The app works, but without caching.")

    with st.spinner("Loading knowledge base..."):
        total_chunks = count_chunks()
        sources = list_sources() if total_chunks else {}

    st.metric("Knowledge chunks", total_chunks)
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for name, chunks_count in sources.items():
                st.write(f"- {name} ({chunks_count})")

    st.divider()
    st.subheader("Session monitor")
    questions = metrics["questions"]
    st.write(f"Questions asked: **{questions}**")
    st.write(f"Answered from cache: **{metrics['cache_hits']}**")
    st.write(f"Approved / not validated: **{metrics['approved']} / {metrics['unvalidated']}**")
    live = questions - metrics["cache_hits"]
    if live > 0:
        st.write(f"Avg iterations (live): **{metrics['iterations'] / live:.1f}**")
    if questions:
        st.write(f"Avg response time: **{metrics['seconds'] / questions:.1f}s**")
    redis_stats = cache.get_stats()
    st.caption(
        f"Cache lookups: {redis_stats['hits']} hits, {redis_stats['misses']} misses, "
        f"{redis_stats['errors']} errors"
    )

    st.divider()
    st.subheader("Agent memories (this session)")
    st.caption(f"Generator: {generator.memory_summary()}")
    st.caption(f"Evaluator: {evaluator.memory_summary()}")
    if st.button("Clear answer cache"):
        cache.bump_knowledge_version()
        st.rerun()
    if st.button("Reset both memories"):
        generator.reset_memory()
        evaluator.reset_memory()
        st.session_state.last_result = None
        st.rerun()


# asking a question
def update_metrics(result):
    metrics["questions"] += 1
    metrics["seconds"] += result["timings"]["total_s"]
    if result["from_cache"]:
        metrics["cache_hits"] += 1
        return
    metrics["iterations"] += result["iterations"]
    if result["status"] == "approved":
        metrics["approved"] += 1
    elif result["status"] == "unvalidated":
        metrics["unvalidated"] += 1


def ask_question():
    question = st.text_area("Your question", placeholder="Ask something about your sources...")
    use_history = st.checkbox(
        "This is a follow-up question (use the Generator's conversation memory)",
        value=False,
    )
    if not st.button("Get answer", type="primary"):
        return

    if count_chunks() == 0:
        st.warning("The knowledge base is empty. Please add at least one source first.")
        return

    try:
        with st.spinner("Generating and evaluating the answer..."):
            result = run_workflow(question, generator, evaluator, use_history=use_history)
        st.session_state.last_result = result
        update_metrics(result)
    except ValueError as error:           # for example: empty question
        st.warning(str(error))
    except WorkflowError as error:        # the language model did not answer
        st.error(str(error))
    except Exception:
        logger.exception("Unexpected error in the workflow")
        st.error("Something went wrong. Please try again. Details are in logs/app.log.")


def show_result(result):
    st.subheader("Answer")
    st.write(result["answer"])

    if result["status"] == "approved":
        st.success(result["message"])
    elif result["status"] == "unvalidated":
        st.warning(result["message"])
    else:
        st.info(result["message"])

    st.caption(
        f"Iterations: {result['iterations']}/{MAX_ITERATIONS} | "
        f"From cache: {'yes' if result['from_cache'] else 'no'} | "
        f"Total time: {result['timings']['total_s']}s"
    )

    if result["sources"]:
        with st.expander("Sources used"):
            for item in result["sources"]:
                st.write(f"- {item['source']} (relevance {item['score']:.2f})")

    if result["history"]:
        st.subheader("Evaluation history")
        for item in result["history"]:
            label = "Approved" if item["approved"] else "Needs improvement"
            with st.expander(f"Iteration {item['iteration']}: {label}"):
                st.markdown("**Answer**")
                st.write(item["answer"])
                st.markdown("**Evaluator feedback**")
                st.write(item["feedback"])
                if item["scores"]:
                    st.markdown("**Scores (1-5)**")
                    st.table([{"criterion": k, "score": v} for k, v in item["scores"].items()])
                if item["unsupported_claims"]:
                    st.markdown("**Unsupported claims**")
                    for claim in item["unsupported_claims"]:
                        st.write(f"- {claim}")


with st.sidebar:
    sidebar()

st.title("🤖 Evaluator-Generator Knowledge Assistant")
st.write(
    "Add knowledge, then ask questions. A Generator writes the answer, an Evaluator "
    f"checks it, and the answer is improved for up to {MAX_ITERATIONS} rounds."
)

st.header("1. Add knowledge")
tab_files, tab_web, tab_wiki = st.tabs(["Files", "Web page", "Wikipedia"])
with tab_files:
    files_tab()
with tab_web:
    web_tab()
with tab_wiki:
    wikipedia_tab()

st.divider()
st.header("2. Ask a question")
ask_question()

if st.session_state.last_result:
    show_result(st.session_state.last_result)
