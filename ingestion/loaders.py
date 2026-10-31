"""Entry point of the ingestion: load_file() chooses the right loader.

    documents.py   text, code, PDF, DOCX, PowerPoint
    audio.py       WAV to text
    web.py         web pages and Wikipedia 
"""
import logging
from pathlib import Path

from ingestion.audio import load_audio
from ingestion.documents import load_code, load_docx, load_pdf, load_powerpoint, load_text

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = [".txt", ".md"]
CODE_EXTENSIONS = [
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cc",
    ".cs", ".go", ".php", ".rb", ".rs", ".kt", ".swift", ".scala", ".html",
    ".css", ".json", ".sql", ".sh", ".bat", ".yaml", ".yml", ".xml", ".ini",
    ".toml", ".csv", ".r", ".m", ".lua", ".ipynb",
]

# which function loads which file type
FILE_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".ppt": load_powerpoint,
    ".pptx": load_powerpoint,
    ".wav": load_audio,
}
for extension in TEXT_EXTENSIONS:
    FILE_LOADERS[extension] = load_text
for extension in CODE_EXTENSIONS:
    FILE_LOADERS[extension] = load_code

SUPPORTED_EXTENSIONS = sorted(FILE_LOADERS)


def load_file(file_path, display_name=None):
    path = Path(file_path)
    extension = path.suffix.lower()
    source = display_name or path.name

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        raise ValueError(f"{source} is {size_mb:.1f} MB; the limit is {50} MB.")
    if path.stat().st_size == 0:
        raise ValueError(f"{source} is empty.")

    logger.info("Loading file %s (%s, %.2f MB)", source, extension, size_mb)

    if extension not in FILE_LOADERS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    documents = FILE_LOADERS[extension](path, source)
    if not documents:
        raise ValueError(f"No readable text was found in {source}.")
    return documents
