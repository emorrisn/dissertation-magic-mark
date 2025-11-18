from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir


def convert_pdf_to_txt(pdf_path: Path, txt_path: Path) -> None:
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pdfminer.six is required to convert PDFs. Install via 'pip install pdfminer.six'."
        ) from e
    text = extract_text(str(pdf_path)) or ""
    ensure_dir(txt_path.parent)
    txt_path.write_text(text, encoding="utf-8")


def convert_docx_to_txt(docx_path: Path, txt_path: Path) -> None:
    try:
        import docx2txt  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "docx2txt is required to convert DOCX. Install via 'pip install docx2txt'."
        ) from e
    text = docx2txt.process(str(docx_path)) or ""
    ensure_dir(txt_path.parent)
    txt_path.write_text(text, encoding="utf-8")
