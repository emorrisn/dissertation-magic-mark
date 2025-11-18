from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

# Dirs
DATA_DIR = Path("data")
EVAL_DIR = Path("../../Evaluation/data")

# Naming patterns
MARKSCHEME_FILENAME_PATTERN = re.compile(r"^m(?P<session>\d+)_A(?P<artifact>\d+)\.(?P<ext>[^.]+)$", re.IGNORECASE)
STUDENT_IMG_PATTERN = re.compile(r"^m(?P<session>\d+)_s(?P<student>\d+)_p(?P<page>\d+)$", re.IGNORECASE)
STUDENT_TXT_PATTERN = re.compile(r"^m(?P<session>\d+)_s(?P<student>\d+)\.(?P<ext>txt|text)$", re.IGNORECASE)


IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif"}
TEXT_EXTS = {"txt", "text"}
PDF_EXTS = {"pdf"}
DOCX_EXTS = {"docx"}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_text_safely(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(encoding="cp1252", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Failed to read text file: {path.name}: {e}")


def parse_markscheme_name(path: Path) -> Optional[Tuple[str, str, str]]:
    m = MARKSCHEME_FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    return m.group("session"), m.group("artifact"), m.group("ext").lower()


def parse_student_img_name(path: Path) -> Optional[Tuple[str, str, str]]:
    m = STUDENT_IMG_PATTERN.match(path.stem)
    if not m:
        return None
    return m.group("session"), m.group("student"), m.group("page")


def is_image(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in IMAGE_EXTS


def is_pdf(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in PDF_EXTS


def is_docx(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in DOCX_EXTS


def is_text(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in TEXT_EXTS


def retrieve_markscheme_files() -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}

    ## Retrieve markscheme files
    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_markscheme_name(path)
        if not parsed:
            continue
        session, artifact, ext = parsed
        grouped.setdefault(session, []).append(path)

    ## Sort them
    for session, paths in grouped.items():
        paths.sort(key=lambda p: int(parse_markscheme_name(p)[1]))
        artifact_ids = [parse_markscheme_name(p)[1] for p in paths]
        print(f"[session m{session}] artifacts: {', '.join(f'A{a}' for a in artifact_ids)}")

    return grouped


# ---- File ops --------------------------------------------------------------
def move_safely(src: Path, dst_dir: Path) -> Path:
    """Move file to dst_dir avoiding name collisions by suffixing __N before extension."""
    ensure_dir(dst_dir)
    target = dst_dir / src.name
    if not target.exists():
        return src.rename(target)
    stem, suffix = src.stem, src.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return src.rename(candidate)
        i += 1


# ---- Content extraction ----------------------------------------------------
def extract_pdf_text(pdf_path: Path) -> str | None:
    try:
        # Try preferred import path first
        from pdfminer_high_level import extract_text  # type: ignore
    except Exception:
        try:
            from pdfminer.high_level import extract_text  # type: ignore
        except Exception:
            print(f"[pdf] Missing dependency pdfminer.six for {pdf_path.name}; skip conversion.")
            return None
    try:
        return extract_text(str(pdf_path)) or ""
    except Exception as e:
        print(f"[pdf] ERROR converting {pdf_path.name}: {e}")
        return None


def extract_docx_text(docx_path: Path) -> str | None:
    try:
        import docx2txt  # type: ignore
    except Exception:
        print(f"[docx] Missing dependency docx2txt for {docx_path.name}; skip conversion.")
        return None
    try:
        return docx2txt.process(str(docx_path)) or ""
    except Exception as e:
        print(f"[docx] ERROR converting {docx_path.name}: {e}")
        return None