from knowledge import cache
from knowledge.vector_store import search_knowledge


def retrieve(query):
    """Retrieve relevant chunks, using the Redis retrieval cache."""
    key = cache.normalize_question(query)
    cached = cache.cache_get("ret", key)
    if cached is not None:
        return cached
    chunks = search_knowledge(query)
    cache.cache_set("ret", chunks, key)
    return chunks


def format_context(chunks):
    """Join the chunks into the text block that is given to the LLMs."""
    return "\n\n".join(f"[Source: {c['source']}]\n{c['content']}" for c in chunks)


def merge_chunks(base, extra, limit):
    merged = list(base)
    seen = {chunk["content"] for chunk in merged}
    for chunk in extra:
        if len(merged) >= limit:
            break
        if chunk["content"] not in seen:
            merged.append(chunk)
            seen.add(chunk["content"])
    return merged


def summarize_sources(chunks):
    best = {}
    for chunk in chunks:
        name = chunk["source"]
        best[name] = max(best.get(name, 0.0), chunk["score"])
    return [{"source": name, "score": score} for name, score in best.items()]
