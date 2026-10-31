"""Small helpers used by all the loaders."""
from pathlib import Path


def finish_documents(documents, source, doc_type, extra=None):
    """Remove empty documents and add the source and type to the metadata."""
    kept = []
    for document in documents:
        if not document.page_content.strip():
            continue
        document.metadata["source"] = source
        document.metadata["type"] = doc_type
        if extra:
            document.metadata.update(extra)
        kept.append(document)
    return kept


def read_text(path):
    """Read a text file. UTF-16 and UTF-8 first, then Arabic Windows (cp1256)."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1256", errors="replace")
