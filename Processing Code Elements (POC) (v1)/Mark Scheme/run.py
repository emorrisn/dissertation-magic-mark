from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FILENAME_PATTERN = re.compile(r"^m(?P<session>\d+)_A(?P<artifact>\d+)\.(?P<ext>[^.]+)$", re.IGNORECASE)


IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif"}
TEXT_EXTS = {"txt", "text"}
PDF_EXTS = {"pdf"}
DOCX_EXTS = {"docx"}


@dataclass(frozen=True)
class ParsedName:
    session_id: str
    artifact_id: str
    ext: str


def parse_filename(path: Path) -> ParsedName | None:
    m = FILENAME_PATTERN.match(path.name)
    if not m:
        return None
    return ParsedName(
        session_id=m.group("session"),
        artifact_id=m.group("artifact"),
        ext=m.group("ext").lower(),
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move_safely(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    target = dst_dir / src.name
    if not target.exists():
        return src.rename(target)
    stem = src.stem
    suffix = src.suffix
    i = 1
    while True:
        candidate = dst_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return src.rename(candidate)
        i += 1


def append_to_aggregate(eval_work_dir: Path, parsed: ParsedName, content: str, source_file: Path) -> None:
    ensure_dir(eval_work_dir)
    aggregate_path = eval_work_dir / f"m{parsed.session_id}.markscheme.txt"
    header = (
        f"\n\n----- BEGIN ARTIFACT A{parsed.artifact_id} ({source_file.name}) -----\n\n"
    )
    footer = f"\n\n----- END ARTIFACT A{parsed.artifact_id} -----\n"
    with aggregate_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write(content)
        f.write(footer)


def convert_pdf_to_txt(pdf_path: Path, txt_path: Path) -> None:
    # Lazy import so we don't require dependency when unused
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pdfminer.six is required to convert PDFs. Install via 'pip install pdfminer.six'."
        ) from e
    text = extract_text(str(pdf_path)) or ""
    txt_path.write_text(text, encoding="utf-8")


def convert_docx_to_txt(docx_path: Path, txt_path: Path) -> None:
    try:
        import docx2txt  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "docx2txt is required to convert DOCX. Install via 'pip install docx2txt'."
        ) from e
    text = docx2txt.process(str(docx_path)) or ""
    txt_path.write_text(text, encoding="utf-8")


def process_pass(base_dir: Path) -> bool:
    mark_scheme_dir = base_dir / "Mark Scheme"
    data_dir = mark_scheme_dir / "data"
    image_to_text_dir = base_dir / "Image to Text" / "data"
    eval_work_dir = base_dir / "Evaluate Work" / "data"
    # No processed/ archive per requirements; we will move processed text to Evaluate Work

    ensure_dir(data_dir)
    ensure_dir(image_to_text_dir)
    ensure_dir(eval_work_dir)
    changed = False
    # No archive directory created

    # 1) Route images to OCR input folder (move out of data_dir)
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path)
        if not parsed:
            continue
        if parsed.ext in IMAGE_EXTS:
            print(f"[image] Moving to OCR: {path.name}")
            move_safely(path, image_to_text_dir)
            changed = True

    # 2) Append existing text artifacts into Evaluate Work, then delete the artifact text file
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path)
        if not parsed:
            continue
        if parsed.ext in TEXT_EXTS:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                # As a fallback, try cp1252 then replace
                try:
                    content = path.read_text(encoding="cp1252", errors="replace")
                except Exception as e:
                    print(f"[text] ERROR reading {path.name}: {e}")
                    continue
            print(f"[text] Appending artifact A{parsed.artifact_id} to m{parsed.session_id}.markscheme.txt")
            append_to_aggregate(eval_work_dir, parsed, content, path)
            # Delete the artifact text file so only the aggregate remains in Evaluate Work
            try:
                path.unlink()
                changed = True
                print(f"[text] Deleted artifact after append: {path.name}")
            except Exception as e:
                print(f"[text] WARN could not delete artifact {path.name}: {e}")

    # 3) Convert PDFs/DOCX to .txt (do not append in the same run). Remove originals after conversion/confirmation.
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path)
        if not parsed:
            continue
        if parsed.ext in PDF_EXTS | DOCX_EXTS:
            target_txt = data_dir / f"m{parsed.session_id}_A{parsed.artifact_id}.txt"
            if target_txt.exists():
                print(f"[convert] Skipping; text already exists for {path.name}")
                # Original is no longer needed; delete to avoid re-processing
                try:
                    path.unlink()
                    print(f"[convert] Deleted original (already had text): {path.name}")
                except Exception as e:
                    print(f"[convert] WARN could not delete original {path.name}: {e}")
                else:
                    changed = True
            else:
                try:
                    if parsed.ext in PDF_EXTS:
                        print(f"[pdf] Converting to text: {path.name}")
                        convert_pdf_to_txt(path, target_txt)
                    elif parsed.ext in DOCX_EXTS:
                        print(f"[docx] Converting to text: {path.name}")
                        convert_docx_to_txt(path, target_txt)
                    else:
                        continue
                except Exception as e:
                    print(f"[convert] ERROR converting {path.name}: {e}")
                    continue
                # After successful conversion, delete the original to prevent re-processing
                try:
                    path.unlink()
                    print(f"[convert] Deleted original after conversion: {path.name}")
                except Exception as e:
                    print(f"[convert] WARN could not delete original {path.name}: {e}")
                else:
                    changed = True
            # The generated .txt will be picked up in the next scan and moved/aggregated

    return changed


def process_until_idle(base_dir: Path) -> None:
    # Keep processing passes until a full pass makes no changes.
    while True:
        if not process_pass(base_dir):
            break


def iter_forever(interval: float, base_dir: Path) -> None:
    try:
        while True:
            process_once(base_dir)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopped.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Process Mark Scheme artifacts: route images to OCR, append text to Evaluate Work, and convert PDFs/DOCX to text."
        )
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep scanning in a loop (polling).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds for --watch mode (default: 5)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Base dir is the workspace folder containing 'Mark Scheme', 'Evaluate Work', 'Image to Text'
    mark_scheme_dir = Path(__file__).resolve().parent
    base_dir = mark_scheme_dir.parent

    if args.watch:
        print(f"Watching '{(mark_scheme_dir / 'data').as_posix()}' every {args.interval}s ...")
        iter_forever(args.interval, base_dir)
    else:
        process_until_idle(base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
