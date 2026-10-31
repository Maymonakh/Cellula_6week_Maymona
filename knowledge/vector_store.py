"""The knowledge base: Chroma vector store + embeddings (cached in Redis)."""
import hashlib
import json
import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from knowledge import cache

logger = logging.getLogger(__name__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100
)


class RedisCachedEmbeddings(Embeddings):
    def __init__(self, base, namespace):
        self.base = base
        self.namespace = namespace

    def _embed(self, texts):
        keys = [cache.content_key("emb", self.namespace, text) for text in texts]
        saved = cache.raw_get_many(keys)

        vectors = [json.loads(value) if value else None for value in saved]
        missing = [i for i, vector in enumerate(vectors) if vector is None]

        if missing:
            new_vectors = self.base.embed_documents([texts[i] for i in missing])
            to_save = {}
            for i, vector in zip(missing, new_vectors):
                vectors[i] = vector
                to_save[keys[i]] = json.dumps(vector)
            cache.raw_set_many(to_save, 604800)
        return vectors

    def embed_documents(self, texts):
        return self._embed(list(texts))

    def embed_query(self, text):
        return self._embed([text])[0]


@lru_cache(maxsize=1)
def get_embeddings():
    logger.info("Loading embedding model %s", "sentence-transformers/all-MiniLM-L6-v2")
    base = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    return RedisCachedEmbeddings(base, namespace="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_vector_store():
    return Chroma(
        collection_name="external_knowledge_v2",
        embedding_function=get_embeddings(),
        persist_directory="chroma_db",
        # cosine distance: the relevance score is the cosine similarity (0 to 1)
        collection_metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk):
    """Same source + same text = same id, so the same file is not added twice."""
    raw = f"{chunk.metadata.get('source', '')}\x1f{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _clean_metadata(metadata):
    """Chroma accepts only str, int, float and bool in the metadata."""
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}


def add_documents(documents):

    unique = {}
    for chunk in text_splitter.split_documents(documents):
        if not chunk.page_content.strip():
            continue
        chunk_id = _chunk_id(chunk)
        if chunk_id not in unique:
            chunk.metadata = _clean_metadata(chunk.metadata)
            unique[chunk_id] = chunk

    ids = list(unique)
    if not ids:
        return {"chunks": 0, "new": 0, "existing": 0}

    store = get_vector_store()
    existing = set(store.get(ids=ids)["ids"])
    new_ids = [i for i in ids if i not in existing]

    for start in range(0, len(new_ids), 500): 
        batch = new_ids[start:start + 500]
        store.add_documents([unique[i] for i in batch], ids=batch)

    if new_ids:
        cache.bump_knowledge_version()  # the old cached answers are not valid anymore

    logger.info("Saved %s new chunks (%s already existed)", len(new_ids), len(existing))
    return {"chunks": len(ids), "new": len(new_ids), "existing": len(existing)}


def search_knowledge(query, k=None, threshold=None):

    k = k or 4
    threshold = 0.2 if threshold is None else threshold

    if count_chunks() == 0:
        return []

    results = get_vector_store().similarity_search_with_relevance_scores(query, k=k)
    chunks = [
        {
            "content": document.page_content,
            "source": str(document.metadata.get("source", "unknown")),
            "score": round(float(score), 4),
        }
        for document, score in results
        if score >= threshold
    ]
    chunks.sort(key=lambda item: item["score"], reverse=True)
    logger.info("Found %s of %s chunks above the threshold %.2f", len(chunks), len(results), threshold)
    return chunks


def count_chunks():
    return get_vector_store()._collection.count()


def list_sources():
    """Return {source name: number of chunks}."""
    metadatas = get_vector_store().get(include=["metadatas"])["metadatas"] or []
    counts = {}
    for metadata in metadatas:
        name = str((metadata or {}).get("source", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


