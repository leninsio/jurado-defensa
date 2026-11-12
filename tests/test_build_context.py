"""Tests for scripts/build_context.py."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import build_context


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_build(bundle, tmp_path: Path):
    sources = build_context.load_sources(bundle.config_path)
    out_dir = tmp_path / "data" / "context"
    slides_dir = tmp_path / "data" / "slides"
    manifest = build_context.build_all(
        sources, out_dir=out_dir, slides_dir=slides_dir, dpi=40
    )
    return manifest, out_dir, slides_dir


def test_build_all_writes_the_expected_bundle(source_bundle, tmp_path):
    manifest, out_dir, slides_dir = run_build(source_bundle, tmp_path)

    assert (out_dir / "tesis.txt").is_file()
    assert (out_dir / "tesis_indice.txt").is_file()
    assert (out_dir / "normas_cuam.txt").is_file()
    assert (out_dir / "presentacion.txt").is_file()
    assert (out_dir / "guia_estudio.txt").is_file()
    assert (out_dir / "manifest.json").is_file()

    saved = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert saved["thesis"]["chars"] == manifest["thesis"]["chars"]
    assert saved["presentation"]["slides"] == 2
    assert saved["norms"]["pages"] == 2
    assert saved["slides"]["count"] == 2
    assert sorted(p.name for p in slides_dir.glob("*.png")) == [
        "slide-01.png",
        "slide-02.png",
    ]


def test_thesis_text_keeps_paragraphs_and_tables(source_bundle, tmp_path):
    _, out_dir, _ = run_build(source_bundle, tmp_path)
    text = (out_dir / "tesis.txt").read_text(encoding="utf-8")

    assert "Proponer un sistema de inventario para la empresa de prueba." in text
    assert "Variable | Indicador" in text
    assert "Inventario | Stock disponible" in text
    # The table is rendered in document order, between the surrounding sections.
    assert text.index("Población y Censo") < text.index("Variable | Indicador")


def test_index_line_numbers_point_at_the_heading(source_bundle, tmp_path):
    _, out_dir, _ = run_build(source_bundle, tmp_path)
    lines = (out_dir / "tesis.txt").read_text(encoding="utf-8").splitlines()
    index_lines = (out_dir / "tesis_indice.txt").read_text(encoding="utf-8").splitlines()

    entries = [line for line in index_lines if re.match(r"^\d+\s{2}", line)]
    titles = [re.sub(r"^\d+\s{2}", "", line) for line in entries]
    assert "Objetivo General" in titles
    assert "Población y Censo" in titles
    for entry in entries:
        number, title = re.match(r"^(\d+)\s{2}(.+)$", entry).groups()
        assert lines[int(number) - 1].strip() == title


def test_norms_text_has_page_markers(source_bundle, tmp_path):
    _, out_dir, _ = run_build(source_bundle, tmp_path)
    text = (out_dir / "normas_cuam.txt").read_text(encoding="utf-8")

    assert "UNIDAD II: Normas de estilo" in text
    assert "UNIDAD III: Proyecto factible" in text
    assert "página 1" in text
    assert "UNIDAD II TEG-600.pdf" in text


def test_presentation_has_one_block_per_slide(source_bundle, tmp_path):
    _, out_dir, _ = run_build(source_bundle, tmp_path)
    text = (out_dir / "presentacion.txt").read_text(encoding="utf-8")

    assert "== Diapositiva 1 ==" in text
    assert "Diapositiva 1: El problema" in text
    assert "== Diapositiva 2 ==" in text
    assert text.index("== Diapositiva 1 ==") < text.index("== Diapositiva 2 ==")


def test_study_guide_is_converted_to_readable_text(source_bundle, tmp_path):
    _, out_dir, _ = run_build(source_bundle, tmp_path)
    text = (out_dir / "guia_estudio.txt").read_text(encoding="utf-8")

    assert "Preguntas frecuentes" in text
    assert "El objetivo general se responde en una línea." in text
    assert "<" not in text
    assert "var ignored" not in text


def test_sources_are_never_modified(source_bundle, tmp_path):
    watched = [source_bundle.thesis, *source_bundle.norms, source_bundle.deck, source_bundle.study_guide]
    before = {path: (digest(path), path.stat().st_mtime_ns) for path in watched}

    run_build(source_bundle, tmp_path)

    after = {path: (digest(path), path.stat().st_mtime_ns) for path in watched}
    assert before == after


def test_missing_source_reports_the_path(source_bundle, tmp_path):
    config = json.loads(source_bundle.config_path.read_text(encoding="utf-8"))
    missing = tmp_path / "sources" / "no-existe.pdf"
    config["sources"]["deck"] = str(missing)
    source_bundle.config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(build_context.ContextBuildError) as error:
        build_context.load_sources(source_bundle.config_path)
    assert str(missing) in str(error.value)


def test_build_is_deterministic(source_bundle, tmp_path):
    _, out_dir, _ = run_build(source_bundle, tmp_path)
    first = (out_dir / "tesis.txt").read_bytes()
    _, out_dir_again, _ = run_build(source_bundle, tmp_path)
    assert first == (out_dir_again / "tesis.txt").read_bytes()


def test_each_slide_image_has_a_matching_text_file(source_bundle, tmp_path):
    _, _, slides_dir = run_build(source_bundle, tmp_path)

    images = sorted(slides_dir.glob("slide-*.png"))
    texts = sorted(slides_dir.glob("slide-*.txt"))
    assert images
    assert [p.stem for p in texts] == [p.stem for p in images]


# ---------------------------------------------------------------------------
# Chapter catalog (S20)
# ---------------------------------------------------------------------------

# The shape of the real thesis index (line number, heading), including the
# table caption that arrives glued to its title and the chapter title line.
CHAPTER_INDEX = [
    (171, "INTRODUCCIÓN"),
    (178, "CAPÍTULO I"),
    (179, "EL PROBLEMA"),
    (180, "Planteamiento del Problema"),
    (188, "Interrogantes de la Investigación"),
    (192, "Objetivos de Investigación"),
    (193, "Objetivo General"),
    (195, "Objetivos Específicos"),
    (199, "Justificación de la Investigación"),
    (202, "CAPÍTULO II"),
    (203, "MARCO TEÓRICO"),
    (204, "Reseña Histórica de la Empresa"),
    (326, "Cuadro 1\nMatriz de Operacionalización de Variables"),
    (338, "CAPÍTULO III"),
    (339, "MARCO METODOLÓGICO"),
    (341, "Tipo y Diseño de la Investigación"),
    (837, "CAPÍTULO V"),
    (838, "CONCLUSIONES Y RECOMENDACIONES"),
    (839, "Conclusiones"),
    (845, "Recomendaciones"),
    (852, "REFERENCIAS"),
    (878, "ANEXO A: INSTRUMENTO DE RECOLECCIÓN DE DATOS"),
]


def test_chapters_are_parsed_from_the_thesis_index():
    chapters = build_context.parse_chapters(CHAPTER_INDEX, total_lines=900)
    by_id = {chapter["id"]: chapter for chapter in chapters}

    assert [chapter["id"] for chapter in chapters] == [
        "introduccion",
        "cap1",
        "cap2",
        "cap3",
        "cap5",
    ]
    assert by_id["cap1"]["label"] == "Capítulo I · El problema"
    assert by_id["cap1"]["start_line"] == 178
    assert by_id["cap1"]["end_line"] == 201
    assert by_id["cap1"]["sections"] == [
        {"title": "Planteamiento del Problema", "start_line": 180},
        {"title": "Interrogantes de la Investigación", "start_line": 188},
        {"title": "Objetivos de Investigación", "start_line": 192},
        {"title": "Objetivo General", "start_line": 193},
        {"title": "Objetivos Específicos", "start_line": 195},
        {"title": "Justificación de la Investigación", "start_line": 199},
    ]
    assert by_id["cap2"]["label"] == "Capítulo II · Marco teórico"
    assert by_id["cap2"]["end_line"] == 337
    # A table caption is not a section of the chapter.
    assert by_id["cap2"]["sections"] == [
        {"title": "Reseña Histórica de la Empresa", "start_line": 204}
    ]
    assert by_id["introduccion"] == {
        "id": "introduccion",
        "label": "Introducción",
        "start_line": 171,
        "end_line": 177,
        "sections": [],
    }
    # CAPÍTULO V already covers the conclusions: no duplicate scope.
    assert "conclusiones" not in by_id
    assert by_id["cap5"]["end_line"] == 851
    assert by_id["cap5"]["sections"] == [
        {"title": "Conclusiones", "start_line": 839},
        {"title": "Recomendaciones", "start_line": 845},
    ]


def test_conclusions_pseudo_chapter_only_without_a_matching_chapter():
    headings = [
        (300, "CONCLUSIONES Y RECOMENDACIONES"),
        (301, "Conclusiones"),
        (305, "Recomendaciones"),
        (320, "REFERENCIAS"),
    ]

    chapters = build_context.parse_chapters(headings, total_lines=340)

    assert chapters == [
        {
            "id": "conclusiones",
            "label": "Conclusiones y recomendaciones",
            "start_line": 300,
            "end_line": 319,
            "sections": [
                {"title": "Conclusiones", "start_line": 301},
                {"title": "Recomendaciones", "start_line": 305},
            ],
        }
    ]


def test_build_writes_the_chapter_catalog(source_bundle, tmp_path):
    manifest, out_dir, _ = run_build(source_bundle, tmp_path)

    catalog = json.loads(
        (out_dir / "capitulos.json").read_text(encoding="utf-8")
    )
    assert catalog == []
    assert manifest["chapters"] == {
        "path": "capitulos.json",
        "count": 0,
    }
