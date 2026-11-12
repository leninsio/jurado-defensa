"""Build the local context bundle for the jury simulator.

Reads the read-only sources declared in ``config.json`` (thesis DOCX, CUAM norm
PDFs, defense deck PDF and study guide HTML) and writes plain-text context files
plus one PNG per slide into ``data/`` (gitignored). The sources are never
modified.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.json"

# Chapter catalog (S20): the thesis headings define the practice scopes.
CHAPTER_ROMANS = {"i": "cap1", "ii": "cap2", "iii": "cap3", "iv": "cap4", "v": "cap5"}
CHAPTER_PATTERN = re.compile(r"^capitulo\s+([ivx]+)$")
INTRO_TITLES = {"introduccion"}
CONCLUSION_TITLES = {"conclusiones y recomendaciones"}
END_MARKERS = ("referencias", "bibliografia", "anexo")
CAPTION_PATTERN = re.compile(
    r"^(cuadro|grafico|tabla|figura)\s+\d+\b", re.IGNORECASE
)


def _fold(text: str) -> str:
    """Lowercase and strip accents for heading comparisons."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(ascii_text.lower().split())


def _clean_heading(text: str) -> str:
    return " ".join(text.split())


def _sentence_case(text: str) -> str:
    cleaned = _clean_heading(text).rstrip(" .:;-")
    if not cleaned:
        return ""
    return cleaned[:1].upper() + cleaned[1:].lower()


def _chapter_number(text: str) -> str | None:
    match = CHAPTER_PATTERN.match(_fold(_clean_heading(text)))
    if not match:
        return None
    return match.group(1)


def _sections_between(
    headings: list[tuple[int, str]],
    start: int,
    end: int,
    skip_title_line: int | None,
) -> list[dict[str, object]]:
    sections: list[dict[str, object]] = []
    seen: set[str] = set()
    for line, text in headings:
        if not start < line < end:
            continue
        if skip_title_line is not None and line == skip_title_line:
            continue
        cleaned = _clean_heading(text)
        if not cleaned:
            continue
        # "Cuadro 1\nMatriz de..." is a table caption, not a section.
        if CAPTION_PATTERN.match(_fold(cleaned.split("\n", 1)[0])):
            continue
        title = cleaned.split("\n", 1)[0].strip()
        key = _fold(title)
        if not title or key in seen:
            continue
        seen.add(key)
        sections.append({"title": title, "start_line": line})
    return sections


def _last_chapter_end(
    headings: list[tuple[int, str]], start: int, total_lines: int
) -> int:
    markers = [
        line
        for line, text in headings
        if line > start and _fold(text).startswith(END_MARKERS)
    ]
    return (min(markers) - 1) if markers else max(start, total_lines)


def _pseudo_chapter(
    headings: list[tuple[int, str]],
    titles: set[str],
    chapter_id: str,
    label: str,
    total_lines: int,
) -> dict[str, object] | None:
    found = next(
        (
            (line, text)
            for line, text in headings
            if _fold(_clean_heading(text)) in titles
        ),
        None,
    )
    if found is None:
        return None
    start_line = found[0]
    next_chapter = next(
        (
            line
            for line, text in headings
            if line > start_line and _chapter_number(text)
        ),
        None,
    )
    end_line = (
        next_chapter - 1
        if next_chapter is not None
        else _last_chapter_end(headings, start_line, total_lines)
    )
    sections = _sections_between(headings, start_line, end_line + 1, None)
    return {
        "id": chapter_id,
        "label": label,
        "start_line": start_line,
        "end_line": end_line,
        "sections": sections,
    }


def parse_chapters(
    headings: list[tuple[int, str]], total_lines: int
) -> list[dict[str, object]]:
    """Derive the practice scopes (chapters) from the thesis headings.

    Each ``CAPÍTULO <roman>`` heading starts a chapter that runs until the next
    chapter; its title is the heading immediately after it. ``Introducción``
    and ``Conclusiones y recomendaciones`` become extra pseudo-chapters when
    those headings exist. Table and figure captions never become sections.
    """
    ordered = sorted(headings, key=lambda item: item[0])
    chapter_starts = [
        (line, roman)
        for line, text in ordered
        if (roman := _chapter_number(text))
    ]
    chapters: list[dict[str, object]] = []

    intro = _pseudo_chapter(
        ordered, INTRO_TITLES, "introduccion", "Introducción", total_lines
    )
    if intro is not None:
        chapters.append(intro)

    for index, (start_line, roman) in enumerate(chapter_starts):
        if index + 1 < len(chapter_starts):
            end_line = max(start_line, chapter_starts[index + 1][0] - 1)
        else:
            end_line = max(
                start_line,
                _last_chapter_end(ordered, start_line, total_lines),
            )
        title_line: int | None = None
        title = ""
        for line, text in ordered:
            if start_line < line <= start_line + 3:
                if _chapter_number(text):
                    break
                title_line, title = line, _clean_heading(text)
                break
        label = f"Capítulo {roman.upper()}"
        if title:
            label = f"{label} · {_sentence_case(title)}"
        chapters.append(
            {
                "id": CHAPTER_ROMANS.get(roman, f"cap{roman}"),
                "label": label,
                "start_line": start_line,
                "end_line": end_line,
                "sections": _sections_between(
                    ordered, start_line, end_line + 1, title_line
                ),
            }
        )

    conclusions = _pseudo_chapter(
        ordered,
        CONCLUSION_TITLES,
        "conclusiones",
        "Conclusiones y recomendaciones",
        total_lines,
    )
    if conclusions is not None:
        # A CAPÍTULO already titled "Conclusiones y recomendaciones" covers the
        # same part of the thesis: do not offer the same scope twice.
        chapter_titles = {
            _fold(chapter["label"].split(" · ", 1)[1])
            for chapter in chapters
            if " · " in chapter["label"]
        }
        if not (chapter_titles & CONCLUSION_TITLES):
            chapters.append(conclusions)

    if intro is not None:
        # Keep the catalog ordered by the position in the thesis.
        chapters.sort(key=lambda chapter: chapter["start_line"])
    return chapters



class ContextBuildError(RuntimeError):
    """Raised when a source is missing or a converter fails."""


@dataclass(frozen=True)
class Sources:
    thesis: Path
    norms: tuple[Path, ...]
    deck: Path
    study_guide: Path


def load_sources(config_path: Path | str = DEFAULT_CONFIG) -> Sources:
    """Resolve and validate every source declared in the configuration."""

    config_path = Path(config_path).expanduser().resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContextBuildError(f"config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ContextBuildError(f"invalid config: {config_path}: {exc}") from exc

    raw = config.get("sources") or {}
    missing_keys = [key for key in ("thesis", "deck", "study_guide") if not raw.get(key)]
    if missing_keys:
        raise ContextBuildError(
            f"config {config_path} is missing sources: {', '.join(missing_keys)}"
        )
    norms = raw.get("norms") or []
    if not norms:
        raise ContextBuildError(f"config {config_path} is missing sources: norms")

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise ContextBuildError(f"source not found: {path}")
        return path

    return Sources(
        thesis=resolve(raw["thesis"]),
        norms=tuple(resolve(value) for value in norms),
        deck=resolve(raw["deck"]),
        study_guide=resolve(raw["study_guide"]),
    )


def _run(command: list[str]) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ContextBuildError(f"missing external tool: {command[0]}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace").strip()
        raise ContextBuildError(f"{' '.join(command)} failed: {stderr}")
    return result


def _pdf_pages(path: Path) -> list[str]:
    result = _run(["pdftotext", "-layout", str(path), "-"])
    pages = result.stdout.decode("utf-8", "replace").split("\f")
    while pages and not pages[-1].strip():
        pages.pop()
    return pages


def _is_heading(paragraph) -> bool:
    text = paragraph.text.strip()
    if not (3 <= len(text) <= 90):
        return False
    if text.endswith((".", ",", ";", ":", "?", "!")):
        return False
    if re.search(r"\t\d+$", text):  # table-of-contents entry with a page number
        return False
    runs = [run for run in paragraph.runs if run.text.strip()]
    return bool(runs) and all(run.bold for run in runs)


def _iter_blocks(document):
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _thesis_blocks(path: Path) -> tuple[list[str], list[tuple[int, str]], int]:
    import docx
    from docx.table import Table

    document = docx.Document(str(path))
    lines: list[str] = []
    headings: list[tuple[int, str]] = []
    tables = 0

    for block in _iter_blocks(document):
        if isinstance(block, Table):
            tables += 1
            lines.append(f"[TABLA {tables}]")
            for row in block.rows:
                cells = [re.sub(r"\s+", " ", cell.text).strip() for cell in row.cells]
                lines.append(" | ".join(cells))
            lines.append(f"[/TABLA {tables}]")
            continue

        text = block.text.strip()
        if not text:
            continue
        lines.append(text)
        if _is_heading(block):
            headings.append((len(lines), text))

    return lines, headings, tables


def _build_index(headings: list[tuple[int, str]]) -> str:
    lines = [
        "# Índice de secciones de la tesis",
        "# El número es la línea correspondiente en tesis.txt",
        "",
    ]
    lines += [f"{number}  {title}" for number, title in headings]
    return "\n".join(lines).rstrip() + "\n"


def _norms_text(norms: tuple[Path, ...]) -> tuple[str, int]:
    blocks: list[str] = []
    pages_total = 0
    for path in norms:
        pages = _pdf_pages(path)
        pages_total += len(pages)
        blocks += [f"== {path.name} ==", ""]
        for number, page in enumerate(pages, start=1):
            blocks.append(f"-- {path.name} · página {number} --")
            blocks.append(page.strip("\n").rstrip())
            blocks.append("")
    return "\n".join(blocks).rstrip() + "\n", pages_total


def _deck_pages(deck: Path) -> list[str]:
    """Return each deck page's text without the "n / total" footer counter."""
    pages = []
    for page in _pdf_pages(deck):
        kept = [
            line.rstrip()
            for line in page.splitlines()
            if not re.fullmatch(r"\s*\d+\s*/\s*\d+\s*", line)
        ]
        pages.append("\n".join(kept).strip())
    return pages


def _deck_text(deck: Path) -> tuple[str, int]:
    pages = _deck_pages(deck)
    blocks: list[str] = []
    for number, page in enumerate(pages, start=1):
        blocks += [f"== Diapositiva {number} ==", page, ""]
    return "\n".join(blocks).rstrip() + "\n", len(pages)


def _write_slide_texts(deck: Path, slides_dir: Path) -> None:
    """Write slide-NN.txt beside each slide-NN.png for the web app."""
    slides_dir.mkdir(parents=True, exist_ok=True)
    for number, page in enumerate(_deck_pages(deck), start=1):
        (slides_dir / f"slide-{number:02d}.txt").write_text(page + "\n", encoding="utf-8")


class _HtmlText(HTMLParser):
    SKIP = {"script", "style", "head", "noscript"}
    BLOCK = {
        "p",
        "div",
        "section",
        "article",
        "li",
        "tr",
        "ul",
        "ol",
        "table",
        "blockquote",
        "br",
        "td",
        "th",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if re.fullmatch(r"h[1-6]", tag):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if re.fullmatch(r"h[1-6]", tag) or tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_text(path: Path) -> str:
    parser = _HtmlText()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()

    lines: list[str] = []
    for raw_line in "".join(parser.parts).splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _slide_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ContextBuildError(f"unexpected slide file name: {path.name}")
    return int(match.group(1))


def _render_slides(deck: Path, slides_dir: Path, dpi: int) -> list[str]:
    slides_dir.mkdir(parents=True, exist_ok=True)
    for stale in slides_dir.glob("slide-*.png"):
        stale.unlink()

    _run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            str(deck),
            str(slides_dir / "slide"),
        ]
    )

    rendered = sorted(slides_dir.glob("slide-*.png"), key=_slide_number)
    names: list[str] = []
    for path in rendered:
        target = slides_dir / f"slide-{_slide_number(path):02d}.png"
        if path != target:
            if target.exists():
                target.unlink()
            path.rename(target)
        names.append(target.name)
    return sorted(names)


def build_all(
    sources: Sources,
    out_dir: Path | str,
    slides_dir: Path | str,
    dpi: int = 110,
) -> dict:
    """Write the whole context bundle and return its manifest."""

    out_dir = Path(out_dir)
    slides_dir = Path(slides_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thesis_lines, headings, table_count = _thesis_blocks(sources.thesis)
    thesis_text = "\n".join(thesis_lines).rstrip() + "\n"
    index_text = _build_index(headings)
    chapters = parse_chapters(headings, len(thesis_lines))
    norms_text, norms_pages = _norms_text(sources.norms)
    presentation_text, slide_count = _deck_text(sources.deck)
    guide_text = _html_text(sources.study_guide)
    slides = _render_slides(sources.deck, slides_dir, dpi)
    _write_slide_texts(sources.deck, slides_dir)

    (out_dir / "tesis.txt").write_text(thesis_text, encoding="utf-8")
    (out_dir / "tesis_indice.txt").write_text(index_text, encoding="utf-8")
    (out_dir / "capitulos.json").write_text(
        json.dumps(chapters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "normas_cuam.txt").write_text(norms_text, encoding="utf-8")
    (out_dir / "presentacion.txt").write_text(presentation_text, encoding="utf-8")
    (out_dir / "guia_estudio.txt").write_text(guide_text, encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "thesis": str(sources.thesis),
            "norms": [str(path) for path in sources.norms],
            "deck": str(sources.deck),
            "study_guide": str(sources.study_guide),
        },
        "thesis": {
            "path": "tesis.txt",
            "chars": len(thesis_text),
            "lines": len(thesis_lines),
            "tables": table_count,
        },
        "index": {"path": "tesis_indice.txt", "sections": len(headings)},
        "chapters": {"path": "capitulos.json", "count": len(chapters)},
        "norms": {
            "path": "normas_cuam.txt",
            "pages": norms_pages,
            "chars": len(norms_text),
        },
        "presentation": {
            "path": "presentacion.txt",
            "slides": slide_count,
            "chars": len(presentation_text),
        },
        "study_guide": {"path": "guia_estudio.txt", "chars": len(guide_text)},
        "slides": {"dir": str(slides_dir), "count": len(slides), "files": slides},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", help="context output directory (default: data/context)")
    parser.add_argument("--slides", help="slide image directory (default: data/slides)")
    parser.add_argument("--dpi", type=int, help="slide image resolution (default: 110)")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).expanduser().read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else REPO_ROOT / config["context"]["dir"]
    slides_dir = Path(args.slides) if args.slides else REPO_ROOT / config["slides"]["dir"]
    dpi = args.dpi if args.dpi else config["slides"].get("dpi", 110)

    try:
        manifest = build_all(load_sources(args.config), out_dir, slides_dir, dpi)
    except ContextBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"context written to {out_dir}")
    print(f"  thesis: {manifest['thesis']['chars']} chars, {manifest['thesis']['lines']} lines")
    print(f"  index: {manifest['index']['sections']} sections")
    print(f"  chapters: {manifest['chapters']['count']}")
    print(f"  norms: {manifest['norms']['pages']} pages")
    print(f"  presentation: {manifest['presentation']['slides']} slides")
    print(f"  study guide: {manifest['study_guide']['chars']} chars")
    print(f"  slides: {manifest['slides']['count']} images in {slides_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
