"""Web pages and Wikipedia."""
import logging
import re
from urllib.parse import unquote, urlparse

from langchain_core.documents import Document

from ingestion.common import finish_documents

logger = logging.getLogger(__name__)


def check_url(url):
    """Accept only http/https links that are not local addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Please provide a valid HTTP or HTTPS URL.")
    if parsed.hostname == "localhost" or parsed.hostname.startswith(("127.", "192.168.", "10.", "0.")):
        raise ValueError("URLs that point to private or local addresses are not allowed.")


def load_web_page(url):
    from langchain_community.document_loaders import WebBaseLoader

    url = url.strip()
    check_url(url)
    logger.info("Loading web page %s", url)

    loader = WebBaseLoader(
        web_paths=[url],
        header_template={"User-Agent": "Mozilla/5.0 (compatible; KnowledgeAssistant/1.0)"},
        requests_kwargs={"timeout": 15},
    )
    documents = finish_documents(loader.load(), url, "web")
    if not documents:
        raise ValueError("The web page contains no readable text.")
    return documents


# a Wikipedia link looks like https://en.wikipedia.org/wiki/Title
WIKI_LINK = re.compile(r"https?://([a-z\-]+)\.(?:m\.)?wikipedia\.org/wiki/([^#?]+)", re.IGNORECASE)


def parse_wikipedia_input(text, default_language="en"):
    """The user can write a title or a full link. Returns (language, title)."""
    text = text.strip()
    match = WIKI_LINK.match(text)
    if match:
        return match.group(1).lower(), unquote(match.group(2)).replace("_", " ")
    return default_language, text


def load_wikipedia_page(title_or_url, language="en"):
    import wikipedia

    language, title = parse_wikipedia_input(title_or_url, language)
    if not title:
        raise ValueError("Please enter a Wikipedia page title or URL.")

    logger.info("Loading Wikipedia page '%s' (%s)", title, language)
    wikipedia.set_lang(language)

    try:
        page = wikipedia.page(title, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as error:
        options = ", ".join(error.options[:5])
        raise ValueError(
            f"'{title}' is ambiguous. Please be more specific, for example: {options}"
        ) from error
    except wikipedia.exceptions.PageError as error:
        raise ValueError(f"No Wikipedia page was found for '{title}'.") from error

    documents = finish_documents(
        [Document(page_content=page.content, metadata={"title": page.title})],
        page.url,
        "wikipedia",
    )
    if not documents:
        raise ValueError("The Wikipedia page is empty.")
    return documents
