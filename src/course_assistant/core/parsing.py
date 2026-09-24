"""Document parsing: file -> ordered pages with text and a rendered image.

Supported input formats and how they are handled:

- ``.pdf``      -> PyMuPDF: extract text per page and render each page to PNG.
- ``.pptx``     -> converted to PDF via headless LibreOffice (``soffice``),
                   then treated as a PDF. Requires LibreOffice installed
                   (``brew install --cask libreoffice``). If ``soffice`` is
                   unavailable we degrade to python-pptx *text* extraction only
                   (no slide images) and surface a warning.
- ``.docx``     -> python-docx text per paragraph (whole doc = one page).
- ``.txt``/``.md`` -> read as text.
- everything else -> rejected with a clear message of supported types.

Images are rendered at a fixed zoom so they are large enough for vision models
and for the visual embedding index.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..config.settings import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md"}
RENDER_ZOOM = 2.0  # 144 dpi


@dataclass
class Page:
    page_num: int
    text: str
    image_path: str | None = None


@dataclass
class ParsedDoc:
    name: str
    sha256: str
    pages: list[Page] = field(default_factory=list)

    def slide_count(self) -> int:
        return len(self.pages)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_soffice() -> str | None:
    return shutil.which("soffice")


def docx_to_text(path: Path) -> str:
    # python-docx for .docx to text
    try:
        from docx import Document  # type: ignore
    except Exception:
        return _extract_plaintext_fallback(path)
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_plaintext_fallback(path: Path) -> str:
    import zipfile

    try:
        with zipfile.ZipFile(path) as z:
            import re
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
            xml = re.sub(r"<w:p[ >]", "\n", xml)
            return re.sub(r"<[^>]+>", "", xml)
    except Exception:
        return ""


def pptx_text_fallback(path: Path) -> list[str]:
    """Text-per-slide via python-pptx (used only if soffice is missing)."""
    from pptx import Presentation  # type: ignore

    prs = Presentation(str(path))
    slides = []
    for slide in prs.slides:
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    texts.append(t)
        slides.append("\n".join(texts))
    return slides


def pdf_pages(pdf_path: Path, out_dir: Path, zoom: float = RENDER_ZOOM):
    """Yield (page_num, text, image_path) for each page of a single PDF."""
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"p{i:03d}.png"
        pix.save(str(img_path))
        pages.append(Page(page_num=i, text=text, image_path=str(img_path)))
    doc.close()
    return pages


def pptx_to_pdf(pptx_path: Path, out_dir: Path) -> Path:
    # Render once per source file: cache the PDF keyed by sha256 so repeated
    # ingests (e.g. many test runs) skip LibreOffice.
    cache_dir = Path(os.environ.get("COURSE_PDF_CACHE",
                     Path(tempfile.gettempdir()) / "mbax6418_pdf_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (sha256_of(pptx_path) + ".pdf")
    out = out_dir / (pptx_path.stem + ".pdf")
    if cached.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, out)
        return out
    soffice = find_soffice()
    if not soffice:
        raise RuntimeError(
            "LibreOffice (soffice) not found. Install it with "
            "`brew install --cask libreoffice` to convert slides to images."
        )
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir", td, str(pptx_path)],
            check=True, capture_output=True, timeout=300,
        )
        pdf = Path(td) / (pptx_path.stem + ".pdf")
        if not pdf.exists():
            raise RuntimeError("LibreOffice produced no PDF")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, cached)
        shutil.copy(pdf, out)
        return out


def parse_file(src: Path, settings: Settings, warn_cb=None) -> ParsedDoc:
    """Parse a single uploaded file into a :class:`ParsedDoc` of pages."""
    ext = Path(src).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    name = Path(src).name
    sha = sha256_of(src)
    include_base = Path(settings.images_dir) / sha[:12]
    include_base.mkdir(parents=True, exist_ok=True)

    ff = None  # optional headless-converted pdf to delete afterward
    try:
        if ext == ".pdf":
            pages = pdf_pages(src, include_base)
        elif ext == ".pptx":
            if not find_soffice():
                texts = pptx_text_fallback(src)
                pages = [Page(page_num=i, text=t) for i, t in enumerate(texts, 1)]
                if warn_cb:
                    warn_cb("Text only: LibreOffice is not installed, so no slide "
                            "images were made and visual questions will not work "
                            "for this deck. Install LibreOffice, then remove and "
                            "re-add it.")
            else:
                pdf = pptx_to_pdf(src, include_base)
                pages = pdf_pages(pdf, include_base)
        elif ext == ".docx":
            text = docx_to_text(src)
            pages = [Page(page_num=1, text=text)]
        else:  # .txt / .md
            text = Path(src).read_text(errors="replace")
            pages = [Page(page_num=1, text=text)]
        return ParsedDoc(name=name, sha256=sha, pages=pages)
    finally:
        if ff is not None:
            try:
                ff.unlink(missing_ok=True)
            except Exception:
                pass
