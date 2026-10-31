"""Loaders for text, code, PDF, DOCX and PowerPoint files.

All of them have the same form: load_x(path, source) -> list of Documents.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from ingestion.common import finish_documents, read_text


def load_text(path, source):
    return finish_documents([Document(page_content=read_text(path))], source, "text")


def load_code(path, source):
    return finish_documents([Document(page_content=read_text(path))], source, "code")


def load_pdf(path, source):
    from langchain_community.document_loaders import PyPDFLoader

    documents = finish_documents(PyPDFLoader(str(path)).load(), source, "pdf")
    if not documents:
        raise ValueError(
            f"{source} has no extractable text. It may be a scanned PDF "
            "(images only); run OCR on it first."
        )
    return documents


def load_docx(path, source):
    from langchain_community.document_loaders import Docx2txtLoader

    return finish_documents(Docx2txtLoader(str(path)).load(), source, "docx")


def convert_ppt_to_pptx(path, output_dir):
    """Old .ppt files are converted with LibreOffice (it must be installed)."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise ValueError(
            "Legacy .ppt files need LibreOffice to be converted. "
            "Please save the file as .pptx and upload it again."
        )
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pptx", "--outdir", output_dir, str(path)],
        check=True,
        capture_output=True,
        timeout=180,
    )
    converted = Path(output_dir) / (Path(path).stem + ".pptx")
    if not converted.exists():
        raise ValueError("LibreOffice could not convert the .ppt file.")
    return converted


def load_powerpoint(path, source):
    from pptx import Presentation

    path = Path(path)
    with tempfile.TemporaryDirectory() as temp_dir:
        if path.suffix.lower() == ".ppt":
            path = convert_ppt_to_pptx(path, temp_dir)

        documents = []
        for number, slide in enumerate(Presentation(str(path)).slides, start=1):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    parts.append(shape.text_frame.text.strip())
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    parts.append(f"Speaker notes: {notes}")
            if parts:
                documents.append(
                    Document(
                        page_content=f"Slide {number}:\n" + "\n".join(parts),
                        metadata={"slide": number},
                    )
                )
    return finish_documents(documents, source, "presentation")
