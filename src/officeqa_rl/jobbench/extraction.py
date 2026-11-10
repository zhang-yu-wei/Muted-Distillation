"""Turn an episode's deliverables into the text and images the judge reads.

A port of the extraction half of JobBench's ``eval/judge.py``. The constants
(per-file character cap, SQLite row cap, image cap, the pattern that decides
which rubric items get images attached) match upstream so scores stay
comparable.

One deliberate difference: where upstream reads a format with exactly one
library, this module falls back to a second one if the first is missing
(``mammoth`` then ``python-docx`` for ``.docx``; ``pdfplumber`` then ``pypdf``
for ``.pdf``). Upstream substitutes an ``[ERROR: ... not available]`` string,
which silently scores every rubric item about that deliverable as failed — a
broken install would read as a bad policy. When the preferred library is
installed the output is upstream's.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import json
import re
import sqlite3
import struct
from pathlib import Path
from zipfile import BadZipFile, ZipFile

MAX_CHARS_PER_FILE = 200_000
SQLITE_EXTS = {"db", "sqlite", "sqlite3"}
SQLITE_ROWS_PER_TABLE = 500
MAX_VISION_IMAGES = 8
# Vertex rejects pathological images even when the compressed PNG is small.
# A 27,750 x 30,542 chart observed in a saved rollout was 4 MB on disk but
# expanded to 848 MP and returned INVALID_ARGUMENT.  Omit such an unusable
# attachment while retaining the file marker in the extracted text.
MAX_VISION_IMAGE_PIXELS = 40_000_000

VISION_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "webp"}
VISION_MIME = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

VISUAL_RUBRIC_PATTERN = re.compile(
    r"\b(plot|figure|visualization|visualisation|visualize|visualise|"
    r"heatmap|histogram|scatter ?plot|biplot|diagram|q[- ]?q)\b",
    re.IGNORECASE,
)

PLAIN_TEXT_EXTS = {
    "txt", "md", "csv", "py", "json", "sh", "log", "xml", "html",
    "css", "js", "ts", "yaml", "yml", "ini", "cfg", "conf", "sql",
    "rules", "geojson",
}


class DocumentReaderUnavailableError(RuntimeError):
    """Host-side extraction dependency failure, never a policy score."""


def require_document_readers() -> None:
    """Check the judge's host environment, independently of sandbox packages."""
    missing = []
    for alternatives in (("mammoth", "docx"), ("pdfplumber", "pypdf"),
                         ("pandas",), ("openpyxl",), ("pptx",)):
        for name in alternatives:
            try:
                importlib.import_module(name)
                break
            except ImportError:
                continue
        else:
            missing.append(" or ".join(alternatives))
    if missing:
        raise DocumentReaderUnavailableError(
            "JobBench judge host is missing document readers: " + ", ".join(missing)
        )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as error:
        return f"[ERROR: Failed to read text file: {path.name}: {error}]"


def _excel_to_text(path: Path) -> str:
    try:
        import pandas as pd

        workbook = pd.ExcelFile(str(path))
        parts = []
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(workbook, sheet_name=sheet)
            parts.append(f"=== Sheet: {sheet} ===\n{frame.to_csv(index=False)}")
        return "\n".join(parts)
    except ImportError:
        raise DocumentReaderUnavailableError(f"pandas/openpyxl not available for {path.name}")
    except Exception as error:
        return f"[ERROR: Failed to read Excel {path.name}: {error}]"


def _docx_to_text(path: Path) -> str:
    try:
        import mammoth

        def embedded_image_placeholder(image):
            return {"src": "embedded-image", "alt": f"Embedded image: {image.content_type}"}

        with path.open("rb") as handle:
            result = mammoth.convert_to_markdown(
                handle,
                convert_image=mammoth.images.img_element(embedded_image_placeholder),
            )
        return result.value
    except ImportError:
        pass
    except Exception as error:
        return f"[ERROR: Failed to read DOCX {path.name}: {error}]"
    try:
        import docx  # python-docx

        document = docx.Document(str(path))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for index, table in enumerate(document.tables):
            parts.append(f"=== Table {index + 1} ===")
            for row in table.rows:
                parts.append(",".join(cell.text.replace("\n", " ") for cell in row.cells))
        return "\n".join(parts)
    except ImportError:
        raise DocumentReaderUnavailableError(f"mammoth/python-docx not available for {path.name}")
    except Exception as error:
        return f"[ERROR: Failed to read DOCX {path.name}: {error}]"


def _pdf_to_text(path: Path) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            parts = []
            for index, page in enumerate(pdf.pages):
                parts.append(f"=== Page {index + 1} ===\n{page.extract_text(layout=True) or ''}")
        return "\n".join(parts)
    except ImportError:
        pass
    except Exception as error:
        return f"[ERROR: Failed to read PDF {path.name}: {error}]"
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        parts = [
            f"=== Page {index + 1} ===\n{page.extract_text() or ''}"
            for index, page in enumerate(reader.pages)
        ]
        return "\n".join(parts)
    except ImportError:
        raise DocumentReaderUnavailableError(f"pdfplumber/pypdf not available for {path.name}")
    except Exception as error:
        return f"[ERROR: Failed to read PDF {path.name}: {error}]"


def _sqlite_to_text(path: Path) -> str:
    try:
        connection = sqlite3.connect(str(path))
        cursor = connection.cursor()
        schema = connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL"
        ).fetchall()
        parts = ["=== Schema ==="]
        parts.extend(row[0] for row in schema if row[0])
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            parts.append(f"\n=== Table: {table} ===")
            try:
                total_rows = connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
                rows = cursor.execute(
                    f'SELECT * FROM "{table}" LIMIT {SQLITE_ROWS_PER_TABLE}'
                ).fetchall()
                columns = [description[0] for description in cursor.description]
                parts.append(
                    f"-- total_rows: {total_rows}; shown: {len(rows)} "
                    f"(LIMIT {SQLITE_ROWS_PER_TABLE})"
                )
                parts.append(",".join(columns))
                for row in rows:
                    parts.append(
                        ",".join("" if value is None else str(value) for value in row)
                    )
            except Exception as error:
                parts.append(f"[ERROR reading table {table}: {error}]")
        connection.close()
        return "\n".join(parts)
    except Exception as error:
        return f"[ERROR: Failed to read SQLite {path.name}: {error}]"


def _pptx_to_text(path: Path) -> str:
    try:
        from pptx import Presentation

        presentation = Presentation(str(path))
        parts = []
        for index, slide in enumerate(presentation.slides):
            parts.append(f"=== Slide {index + 1} ===")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    parts.append(shape.text)
        return "\n".join(parts)
    except ImportError:
        raise DocumentReaderUnavailableError(f"python-pptx not available for {path.name}")
    except Exception as error:
        return f"[ERROR: Failed to read PowerPoint {path.name}: {error}]"


def _notebook_to_text(path: Path) -> str:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        parts = []
        for cell in notebook.get("cells", []):
            parts.append(f"=== {cell['cell_type']} ===")
            parts.append("".join(cell.get("source", [])))
            for output in cell.get("outputs", []):
                if "text" in output:
                    parts.append("".join(output["text"]))
        return "\n".join(parts)
    except Exception as error:
        return f"[ERROR: Failed to read notebook {path.name}: {error}]"


def convert_file_to_text(path: Path) -> str:
    """Render one deliverable as text for the judge prompt."""
    ext = path.suffix.lower().lstrip(".")
    if ext in PLAIN_TEXT_EXTS:
        return _read_text(path)
    if ext in ("xlsx", "xls"):
        return _excel_to_text(path)
    if ext == "docx":
        return _docx_to_text(path)
    if ext == "pdf":
        return _pdf_to_text(path)
    if ext in SQLITE_EXTS:
        return _sqlite_to_text(path)
    if ext == "pptx":
        return _pptx_to_text(path)
    if ext == "ipynb":
        return _notebook_to_text(path)
    if ext in ("png", "jpg", "jpeg", "gif", "svg", "bmp"):
        return f"[Image file: {path.name} — cannot extract text content]"
    return f"[Binary or unsupported file type: {ext} — {path.name}]"


def extract_all_file_contents(output_dir: Path) -> str:
    """Concatenate every deliverable under ``output_dir`` as judge input."""
    parts = []
    for file_path in sorted(output_dir.rglob("*")):
        if not file_path.is_file():
            continue
        content = convert_file_to_text(file_path)
        ext = file_path.suffix.lower().lstrip(".")
        if ext not in SQLITE_EXTS and len(content) > MAX_CHARS_PER_FILE:
            content = (
                content[:MAX_CHARS_PER_FILE]
                + f"\n... [Content truncated at {MAX_CHARS_PER_FILE} characters]"
            )
        parts.append(f"=== FILE: {file_path.name} ===\n{content}\n")
    return "\n".join(parts)


def rubric_needs_vision(rubric: dict) -> bool:
    """True when a rubric item asks about something only an image can show."""
    text = rubric.get("rubric", "") or ""
    criterion = rubric.get("criterion", [])
    if isinstance(criterion, list):
        text = text + " " + " ".join(str(entry) for entry in criterion)
    elif isinstance(criterion, str):
        text = text + " " + criterion
    return bool(VISUAL_RUBRIC_PATTERN.search(text))


def _image_data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _oversized_png(payload: bytes) -> bool:
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
        return False
    width, height = struct.unpack(">II", payload[16:24])
    return width * height > MAX_VISION_IMAGE_PIXELS


def collect_image_attachments(
    output_dir: Path, cap: int = MAX_VISION_IMAGES
) -> list[tuple[str, str]]:
    """Collect standalone and embedded images as deduplicated data URLs."""
    if not output_dir.exists() or cap <= 0:
        return []

    attachments: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, mime: str, payload: bytes) -> bool:
        if mime == "image/png" and _oversized_png(payload):
            return False
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            return False
        seen.add(digest)
        attachments.append((name, _image_data_url(mime, payload)))
        return len(attachments) >= cap

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext not in VISION_IMAGE_EXTS:
            continue
        try:
            payload = path.read_bytes()
        except Exception:
            continue
        if add(path.relative_to(output_dir).as_posix(), VISION_MIME[ext], payload):
            return attachments

    for docx_path in sorted(output_dir.rglob("*.docx")):
        try:
            with ZipFile(docx_path) as archive:
                media = [
                    name
                    for name in archive.namelist()
                    if name.startswith("word/media/")
                    and name.rsplit(".", 1)[-1].lower() in VISION_IMAGE_EXTS
                ]
                for media_name in sorted(media):
                    mime = VISION_MIME[media_name.rsplit(".", 1)[-1].lower()]
                    name = f"{docx_path.relative_to(output_dir).as_posix()}:{media_name}"
                    if add(name, mime, archive.read(media_name)):
                        return attachments
        except (BadZipFile, OSError, KeyError):
            continue

    for notebook_path in sorted(output_dir.rglob("*.ipynb")):
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(notebook, dict):
            continue
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            # Agent-produced notebooks are not guaranteed to satisfy the
            # nbformat schema.  A malformed cell/output must not abort the
            # entire episode after its rollout has already completed.
            if not isinstance(cell, dict):
                continue
            for output_index, output in enumerate(cell.get("outputs", [])):
                if not isinstance(output, dict):
                    continue
                data = output.get("data", {})
                if not isinstance(data, dict):
                    continue
                for mime in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                    encoded = data.get(mime)
                    if not encoded:
                        continue
                    if isinstance(encoded, list):
                        encoded = "".join(str(part) for part in encoded)
                    if not isinstance(encoded, str):
                        continue
                    try:
                        payload = base64.b64decode("".join(encoded.split()), validate=True)
                    except (binascii.Error, ValueError):
                        continue
                    name = (
                        f"{notebook_path.relative_to(output_dir).as_posix()}:"
                        f"cell{cell_index}:out{output_index}"
                    )
                    if add(name, mime, payload):
                        return attachments
    return attachments


__all__ = [
    "MAX_CHARS_PER_FILE",
    "MAX_VISION_IMAGES",
    "MAX_VISION_IMAGE_PIXELS",
    "SQLITE_ROWS_PER_TABLE",
    "VISUAL_RUBRIC_PATTERN",
    "collect_image_attachments",
    "convert_file_to_text",
    "extract_all_file_contents",
    "rubric_needs_vision",
]
