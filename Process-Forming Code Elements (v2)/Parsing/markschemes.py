from __future__ import annotations

from pathlib import Path
import re
from typing import List
from scripts.utils import (
    PDF_EXTS,
    DOCX_EXTS,
    TEXT_EXTS,
    IMAGE_EXTS,
    DATA_DIR,
    EVAL_DIR,
    ensure_dir,
    read_text_safely,
    parse_markscheme_name,
    retrieve_markscheme_files,
    move_safely,
    extract_pdf_text,
    extract_docx_text,
)
 

class MarkschemeProcessor:
    def __init__(self, data_dir: Path | None = None, eval_dir: Path | None = None) -> None:
        # Allow override for testing; default to globals from utils
        self.data_dir = data_dir or DATA_DIR
        self.eval_dir = eval_dir or EVAL_DIR

    # --- Main processing ----------------------------------------------------
    def process(self) -> List[str]:
        """Process markscheme artifacts and return ready session IDs.

        Behaviour:
        - Keep an aggregate file per session in data_dir while any artifacts remain.
        - Append any available text artifacts into the aggregate and delete the text artifacts.
        - Convert PDFs/DOCX to text, then append and delete originals.
        - Only when a session has no remaining artifacts in data_dir, move the session aggregate
          from data_dir to eval_dir (using safe move semantics).
        """
        print("Processing markschemes...")
        ensure_dir(self.data_dir)
        ensure_dir(self.eval_dir)

        # Optional retrofit: clean existing aggregates (collapse excessive blank lines)
        self._retro_clean_aggregates()

        # Stage 1: ensure per-session aggregates exist in data_dir
        markschemes = retrieve_markscheme_files()
        self._ensure_aggregates(markschemes)

        # Stage 2: convert DOCX/PDF and append immediately
        markschemes = retrieve_markscheme_files()
        self._convert_and_append_docs(markschemes)

        # Stage 3: append text artifacts and delete them
        markschemes = retrieve_markscheme_files()
        self._append_text_artifacts(markschemes)

        # Stage 4: finalize sessions (move aggregates when no artifacts remain)
        ready_sessions = self._finalize_sessions()
        print(f"Ready sessions: {', '.join(f'm{s}' for s in ready_sessions) or 'None'}")
        return ready_sessions

    # ---- Stages -------------------------------------------------------------
    def _ensure_aggregates(self, markschemes: dict[str, list[Path]]) -> None:
        for session in list(markschemes.keys()):
            aggregate_path = self.data_dir / f"m{session}.markscheme.txt"
            if not aggregate_path.exists():
                try:
                    aggregate_path.touch()
                    print(f"[init] Created aggregate: {aggregate_path.name}")
                except Exception as e:
                    print(f"[init] ERROR creating aggregate for m{session}: {e}")

    def _convert_and_append_docs(self, markschemes: dict[str, list[Path]]) -> None:
        for session, paths in markschemes.items():
            for path in list(paths):
                parsed = parse_markscheme_name(path)
                if not parsed:
                    continue
                _, artifact, ext = parsed
                lower_ext = ext.lower()
                if lower_ext in PDF_EXTS | DOCX_EXTS:
                    aggregate_path = self.data_dir / f"m{session}.markscheme.txt"
                    # Extract content in-memory and append; no temporary files
                    content: str | None
                    try:
                        if lower_ext in PDF_EXTS:
                            content = extract_pdf_text(path)
                        else:
                            content = extract_docx_text(path)
                    except Exception as e:
                        print(f"[convert] ERROR extracting text from {path.name}: {e}")
                        continue
                    if not content and content != "":
                        # extraction failed or returned None; retry next run
                        continue

                    # Clean excessive blank lines and whitespace
                    if content is not None:
                        content = self._clean_text(content)
                    header = f"\n\n# BEGIN ARTIFACT {artifact}\n\n"
                    footer = f"\n\n# END ARTIFACT {artifact}\n\n"
                    try:
                        with aggregate_path.open("a", encoding="utf-8", newline="\n") as f:
                            f.write(header)
                            f.write(content or "")
                            f.write(footer)
                        print(f"[convert+append] Appended A{artifact} from {path.name} -> {aggregate_path.name}")
                    except Exception as e:
                        print(f"[convert+append] ERROR appending {path.name}: {e}")
                        # don't delete original so we can retry next run
                        continue

                    # After successful append, delete original
                    try:
                        path.unlink()
                        paths.remove(path)
                    except Exception as e:
                        print(f"[convert+append] WARN could not delete original {path.name}: {e}")

    def _append_text_artifacts(self, markschemes: dict[str, list[Path]]) -> None:
        for session, paths in markschemes.items():
            text_artifacts = [p for p in paths if p.suffix.lower().lstrip('.') in TEXT_EXTS]
            if not text_artifacts:
                continue
            text_artifacts.sort(key=lambda p: int(parse_markscheme_name(p)[1]))
            aggregate_path = self.data_dir / f"m{session}.markscheme.txt"
            for p in text_artifacts:
                parsed = parse_markscheme_name(p)
                if not parsed:
                    continue
                _, artifact, _ = parsed
                try:
                    content = read_text_safely(p)
                except Exception as e:
                    print(f"[append] ERROR reading {p.name}: {e}")
                    continue
                content = self._clean_text(content)
                header = f"\n\n# BEGIN ARTIFACT A{artifact} ({p.name})\n\n"
                footer = f"\n\n# END ARTIFACT A{artifact}\n\n"
                with aggregate_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(header)
                    f.write(content)
                    f.write(footer)
                print(f"[append] Added A{artifact} to {aggregate_path.name}")
                try:
                    p.unlink()
                    print(f"[cleanup] Deleted {p.name}")
                except Exception as e:
                    print(f"[cleanup] WARN could not delete {p.name}: {e}")

    def _finalize_sessions(self) -> List[str]:
        ready_sessions: List[str] = []
        remaining = retrieve_markscheme_files()
        for session, paths in remaining.items():
            outstanding = [
                p for p in paths
                if p.suffix.lower().lstrip('.') in (IMAGE_EXTS | PDF_EXTS | DOCX_EXTS | TEXT_EXTS)
            ]
            if len(outstanding) == 0:
                ready_sessions.append(session)
                aggregate_in_data = self.data_dir / f"m{session}.markscheme.txt"
                if aggregate_in_data.exists():
                    try:
                        dst = move_safely(aggregate_in_data, self.eval_dir)
                        print(f"[move] Finalized session m{session} -> {dst}")
                    except Exception as e:
                        print(f"[move] ERROR moving aggregate for m{session}: {e}")
        return ready_sessions

    # ---- Text cleaning -----------------------------------------------------
    def _clean_text(self, text: str) -> str:
        # Normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Trim trailing whitespace per line
        lines = [ln.rstrip() for ln in text.split('\n')]
        text = '\n'.join(lines)
        # Collapse runs of >=3 blank lines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove lines that are only whitespace between non-empty lines (already handled by rstrip)
        # Final strip of leading/trailing blank lines (preserve single trailing newline when appended)
        text = text.strip('\n')
        return text

    def _retro_clean_aggregates(self) -> None:
        """Clean already existing aggregate files to remove excessive blank lines.

        This is run each process pass so previously appended content (before cleaning logic existed)
        is normalized. Headers and footers are preserved exactly.
        """
        for agg in self.data_dir.glob("m*.markscheme.txt"):
            try:
                original = agg.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            lines = original.splitlines()
            changed = False
            result: List[str] = []
            blank_run = 0
            for ln in lines:
                if ln.startswith("# BEGIN ARTIFACT") or ln.startswith("# END ARTIFACT"):
                    result.append(ln.rstrip())
                    blank_run = 0
                    continue
                stripped = ln.strip()
                if stripped == "":
                    blank_run += 1
                    if blank_run <= 2:
                        result.append("")
                    else:
                        changed = True
                    continue
                # non blank
                if ln.rstrip() != ln:
                    changed = True
                result.append(ln.rstrip())
                blank_run = 0
            new_content = "\n".join(result).rstrip() + "\n"
            if new_content != original:
                try:
                    agg.write_text(new_content, encoding="utf-8")
                    print(f"[clean] Normalized whitespace in {agg.name}")
                except Exception:
                    pass
