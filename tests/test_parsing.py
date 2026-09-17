"""Unit tests for document parsing / ingestion."""
import pytest

from course_assistant.core.parsing import (SUPPORTED_EXTENSIONS, parse_file,
                                           sha256_of)
from course_assistant.config.settings import Settings


@pytest.fixture(scope="module")
def settings():
    from pathlib import Path
    import tempfile
    td = Path(tempfile.mkdtemp())
    os.environ["COURSE_DATA_DIR"] = str(td)
    s = Settings()
    return s


import os  # noqa: E402


def test_txt_parses_to_one_page(settings, tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello course\nsecond line")
    doc = parse_file(f, settings)
    assert doc.name == "a.txt"
    assert len(doc.pages) == 1
    assert "hello course" in doc.pages[0].text
    assert doc.pages[0].image_path is None


def test_unsupported_extension_raises(settings, tmp_path):
    f = tmp_path / "a.xlsx"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError):
        parse_file(f, settings)


def test_sha256_is_content_hash(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("same")
    assert sha256_of(a) == sha256_of(a)


def test_pptx_parses_with_slide_images(settings):
    """Build a 2-slide pptx and verify LibreOffice conversion yields slide
    images and per-slide text."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    slide1 = prs.slides.add_slide(prs.slide_layouts[1])
    slide1.shapes.title.text = "Hello Slides"
    slide1.placeholders[1].text = "First slide body"
    slide2 = prs.slides.add_slide(prs.slide_layouts[5])
    tb = slide2.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    tb.text = "Second slide about Vibe Coding on Prod"
    pptx = settings.materials_dir / "deck.pptx"
    pptx.parent.mkdir(parents=True, exist_ok=True)
    prs.save(pptx)

    doc = parse_file(pptx, settings)
    assert doc.slide_count() == 2
    assert all(p.image_path for p in doc.pages), "every slide should render an image"
    assert any("Vibe Coding" in p.text for p in doc.pages)
    # rendered images exist on disk
    for p in doc.pages:
        from pathlib import Path
        assert Path(p.image_path).exists()
