"""Shared fixtures and helpers for the jury simulator test suite."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import docx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for entry in (REPO_ROOT, SCRIPTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a minimal multi-page PDF with one text line per page.

    Hand-built so tests do not depend on an external PDF generator; the file is
    valid enough for poppler's pdftotext and pdftoppm.
    """

    objects: dict[int, bytes] = {}
    page_ids = [3 + 2 * i for i in range(len(pages))]
    content_ids = [4 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = b" ".join(b"%d 0 R" % page_id for page_id in page_ids)
    objects[2] = b"<< /Type /Pages /Kids [" + kids + b"] /Count %d >>" % len(pages)

    for page_id, content_id, text in zip(page_ids, content_ids, pages):
        objects[page_id] = (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
            % (font_id, content_id)
        )
        stream = b"BT /F1 24 Tf 72 700 Td (" + text.encode("latin-1") + b") Tj ET"
        objects[content_id] = (
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )

    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    payload = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(payload)
        payload += b"%d 0 obj\n" % object_id + objects[object_id] + b"\nendobj\n"

    xref_offset = len(payload)
    max_id = max(objects)
    payload += b"xref\n0 %d\n" % (max_id + 1)
    payload += b"0000000000 65535 f \n"
    for object_id in range(1, max_id + 1):
        payload += b"%010d 00000 n \n" % offsets[object_id]
    payload += (
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (max_id + 1, xref_offset)
    )

    path.write_bytes(bytes(payload))
    return path


def write_thesis_docx(path: Path) -> Path:
    document = docx.Document()
    document.add_paragraph("SISTEMA DE INVENTARIO DE PRUEBA")
    heading = document.add_paragraph()
    heading.add_run("Objetivo General").bold = True
    document.add_paragraph("Proponer un sistema de inventario para la empresa de prueba.")
    heading = document.add_paragraph()
    heading.add_run("Población y Censo").bold = True
    document.add_paragraph("La población está conformada por cinco trabajadores.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Variable"
    table.cell(0, 1).text = "Indicador"
    table.cell(1, 0).text = "Inventario"
    table.cell(1, 1).text = "Stock disponible"
    document.save(path)
    return path


@dataclass
class SourceBundle:
    root: Path
    config_path: Path
    thesis: Path
    norms: list[Path]
    deck: Path
    study_guide: Path
    config: dict


@pytest.fixture
def source_bundle(tmp_path: Path) -> SourceBundle:
    sources = tmp_path / "sources"
    sources.mkdir()

    thesis = write_thesis_docx(sources / "final.docx")
    norms_ii = write_pdf(sources / "UNIDAD II TEG-600.pdf", ["UNIDAD II: Normas de estilo"])
    norms_iii = write_pdf(sources / "UNIDAD III TEG-600.pdf", ["UNIDAD III: Proyecto factible"])
    deck = write_pdf(
        sources / "presentacion_defensa.pdf",
        ["Diapositiva 1: El problema", "Diapositiva 2: Los objetivos"],
    )
    study_guide = sources / "guia_estudio.html"
    study_guide.write_text(
        "<html><body><h2>Preguntas frecuentes</h2>"
        "<p>El objetivo general se responde en una línea.</p>"
        "<script>var ignored = 1;</script></body></html>",
        encoding="utf-8",
    )

    config = {
        "sources": {
            "thesis": str(thesis),
            "norms": [str(norms_ii), str(norms_iii)],
            "deck": str(deck),
            "study_guide": str(study_guide),
        },
        "slides": {"dpi": 40},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    return SourceBundle(
        root=tmp_path,
        config_path=config_path,
        thesis=thesis,
        norms=[norms_ii, norms_iii],
        deck=deck,
        study_guide=study_guide,
        config=config,
    )
