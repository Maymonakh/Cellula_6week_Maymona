"""WAV audio to text (speech-to-text with faster-whisper)."""
import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from ingestion.common import finish_documents
from knowledge import cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_whisper_model():
    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model '%s'", "small")
    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(path):
    """Return (text, language) of a WAV file."""
    try:
        segments, info = get_whisper_model().transcribe(str(path), beam_size=5, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as error:
        logger.exception("Speech-to-text failed")
        raise ValueError(
            "Speech-to-text failed. If the message mentions 'metadata_errors', "
            "run: pip install \"av<19\". Details: " + str(error)
        ) from error
    return text, info.language


def load_audio(file_path, display_name=None):
    path = Path(file_path)
    source = display_name or path.name

    if path.suffix.lower() != ".wav":
        raise ValueError("Please provide a WAV audio file.")

    # a real WAV file starts with "RIFF" and has "WAVE" at byte 8
    with open(path, "rb") as file:
        header = file.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError(f"{source} is not a valid WAV file.")

    # Speech-to-text is slow, so the transcript is saved in Redis
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    key = cache.content_key("doc", "wav", "small", file_hash)

    saved = cache.raw_get_many([key])[0]
    if saved:
        logger.info("Audio transcript found in the cache for %s", source)
        language, _, text = saved.partition("\n")
    else:
        logger.info("Transcribing %s", source)
        text, language = transcribe(path)
        logger.info("Transcript (%s): %s", language, text[:200])
        if text:
            cache.raw_set_many({key: f"{language}\n{text}"}, 604800)

    if not text:
        raise ValueError("No speech could be extracted from the audio.")

    return finish_documents([Document(page_content=text)], source, "audio", {"language": language})
