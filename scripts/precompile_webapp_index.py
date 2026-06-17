#!/usr/bin/env python3
"""Precompile the main WebApp JSX block from webapp/index.html.

This is intentionally a small bridge, not a frontend migration. The source
remains embedded in index.html so existing static checks keep reading the same
application code, while runtime loads a compiled bundle instead of Babel.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "webapp" / "index.html"
OUT_JS = ROOT / "webapp" / "index.compiled.js"
ESBUILD_PACKAGE = "esbuild@0.25.5"
SOURCE_TYPE = "application/x-extraarena-jsx-source"
SOURCE_ID = "extraarena-main-jsx-source"
ROOT_RENDER_MARKER = "ReactDOM.createRoot(document.getElementById('root')).render(<App/>);"

BABEL_SCRIPT_RE = re.compile(
    r"\n?<script\s+src=\"https://unpkg\.com/@babel/standalone@[^\"<>]+\"\s+crossorigin=\"anonymous\"></script>\n?",
)
COMPILED_SCRIPT_RE = re.compile(
    r"\n?<script\s+src=\"index\.compiled\.js\?v=[0-9a-f]+\"></script>\n?",
)
SCRIPT_OPEN_RE = re.compile(r"<script(?P<attrs>[^>]*)>", re.DOTALL)


@dataclass(frozen=True)
class SourceBlock:
    start: int
    end: int
    body: str


def _script_attrs_match(attrs: str) -> bool:
    return (
        'type="text/babel"' in attrs
        or f'type="{SOURCE_TYPE}"' in attrs
        or f"id=\"{SOURCE_ID}\"" in attrs
    )


def _sanitize_source(source: str) -> str:
    # Repairs a previous failed rewrite without touching executable app logic.
    return re.sub(r'<script\s+id="' + re.escape(SOURCE_ID) + r'"[^>]*>\n?', "\n", source)


def _find_source_script(html: str) -> SourceBlock:
    candidates: list[SourceBlock] = []
    for match in SCRIPT_OPEN_RE.finditer(html):
        if not _script_attrs_match(match.group("attrs")):
            continue
        marker_pos = html.find(ROOT_RENDER_MARKER, match.end())
        if marker_pos == -1:
            continue
        close_pos = html.find("</script>", marker_pos)
        if close_pos == -1:
            raise RuntimeError("Found main JSX source start but no closing </script> after React root render.")
        candidates.append(
            SourceBlock(
                start=match.start(),
                end=close_pos + len("</script>"),
                body=_sanitize_source(html[match.end() : close_pos]),
            )
        )

    if not candidates:
        raise RuntimeError("Could not find the main WebApp JSX source script.")

    return min(candidates, key=lambda block: block.start)


def _compile_jsx(source: str, outfile: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="extraarena-precompile-") as tmp:
        source_file = Path(tmp) / "index.jsx"
        source_file.write_text(source, encoding="utf-8")
        subprocess.run(
            [
                "npx",
                "--yes",
                ESBUILD_PACKAGE,
                str(source_file),
                "--bundle=false",
                "--platform=browser",
                "--format=iife",
                "--target=es2018",
                "--jsx-factory=React.createElement",
                "--jsx-fragment=React.Fragment",
                "--minify",
                "--legal-comments=none",
                f"--outfile={outfile}",
            ],
            cwd=ROOT,
            check=True,
        )


def _bundle_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _rewrite_html(html: str, source_block: SourceBlock, bundle_hash: str) -> str:
    source = source_block.body
    html = BABEL_SCRIPT_RE.sub("\n", html)
    html = COMPILED_SCRIPT_RE.sub("\n", html)

    replacement = (
        f'<script id="{SOURCE_ID}" type="{SOURCE_TYPE}" '
        f'data-compiled-src="index.compiled.js?v={bundle_hash}">{source}</script>\n'
        f'<script src="index.compiled.js?v={bundle_hash}"></script>'
    )
    return html[: source_block.start] + replacement + html[source_block.end :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if index.html or compiled output is stale.")
    args = parser.parse_args()

    html = INDEX_HTML.read_text(encoding="utf-8")
    source_block = _find_source_script(html)
    source = source_block.body

    if args.check:
        with tempfile.TemporaryDirectory(prefix="extraarena-precompile-check-") as tmp:
            candidate_js = Path(tmp) / OUT_JS.name
            _compile_jsx(source, candidate_js)
            bundle_hash = _bundle_hash(candidate_js)
            rewritten = _rewrite_html(html, source_block, bundle_hash)
            html_stale = rewritten != html
            bundle_stale = not OUT_JS.exists() or OUT_JS.read_bytes() != candidate_js.read_bytes()
            if html_stale or bundle_stale:
                if html_stale:
                    print("webapp/index.html is not precompiled or has stale bundle hash.", file=sys.stderr)
                if bundle_stale:
                    print("webapp/index.compiled.js is missing or stale.", file=sys.stderr)
                return 1
            return 0

    _compile_jsx(source, OUT_JS)
    bundle_hash = _bundle_hash(OUT_JS)
    rewritten = _rewrite_html(html, source_block, bundle_hash)

    INDEX_HTML.write_text(rewritten, encoding="utf-8")
    print(f"Wrote {OUT_JS.relative_to(ROOT)}")
    print(f"Bundle hash: {bundle_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
