#!/usr/bin/env python3
"""РУЧНОЙ хелпер синхронизации borrowed-арены (off by default).

Копирует ``webapp/{arena.html,arena.js,arena-styles.css,safe-area.js}`` в
``extra_orchestra/webapp_borrow/`` verbatim, добавляет NOTE-заголовок со
snapshot-коммитом и вшивает ``window.__orchestraInit`` hook в arena.js
(обход prebattle-гейта для моста-плеера).

НЕ запускается автоматически — minimal-port constraint: никогда полный
auto-refresh из webapp/. Запускать осознанно после обновления webapp/ и
обязательно перезаписать snapshot-коммит в HEADER_COMMIT.

Audio-remap НЕ нужен: сервер монтирует ``/DesignAssets/`` целиком, поэтому
оригинальные пути ``/DesignAssets/Sounds/arena/*`` разрешаются напрямую
(arena.js принимает оба префикса — line ~943).

Использование:
    python3 extra_orchestra/scripts/sync_borrowed.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # worktree root
SRC = ROOT / "webapp"
DST = ROOT / "extra_orchestra" / "webapp_borrow"

HEADER_COMMIT = "1282fcb8"  # worktree-NewCards2606 — обновить при ре-снапшоте

FILES = ["arena.html", "arena.js", "arena-styles.css", "safe-area.js"]

NOTE_JS = (
    f"/* ExtraOrchestra borrowed snapshot — source webapp/ @ commit {HEADER_COMMIT}\n"
    f"   (worktree-NewCards2606). DO NOT auto-sync; use\n"
    f"   extra_orchestra/scripts/sync_borrowed.py manually. Minimal-port: never\n"
    f"   full-refresh from webapp/. */\n"
)
NOTE_CSS = NOTE_JS  # CSS и JS используют одинаковый блочный комментарий

# Вшивается в arena.js сразу после блока top-level `let`-ов (~line 27).
# Функция замыкается на лексические top-level bindings и может их мутировать
# (prebattleComplete и др. — это `let`, НЕ свойства window; см. arena.js:17-27).
ORCHESTRA_HOOK = (
    "\n// === ExtraOrchestra baked hook (DO NOT remove with sync) ===\n"
    "window.__orchestraInit = function () {\n"
    "  try {\n"
    "    prebattleRendered = true;\n"
    "    prebattleSequenceStarted = true;\n"
    "    prebattleComplete = true;\n"
    "  } catch (e) { /* top-level lets absent? */ }\n"
    "  return true;\n"
    "};\n"
    "window.__orchestraPresent = true;\n"
)


def _inject_hook(text: str) -> str:
    """Вшить __orchestraInit после строки `let prebattleComplete = ...;`."""
    marker = "let prebattleComplete = "
    idx = text.find(marker)
    if idx == -1:
        # fallback: после `let socketJoinRetryTimer` блока
        idx = text.find("let socketJoinRetryTimer")
        if idx == -1:
            raise RuntimeError("cannot locate top-level lets in arena.js to inject hook")
    end = text.find("\n", idx) + 1
    if "window.__orchestraInit" in text:
        return text  # уже вшит
    return text[:end] + ORCHESTRA_HOOK + text[end:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="перезаписать даже если файлы совпадают")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"[sync] source {SRC} not found", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        src = SRC / name
        dst = DST / name
        if not src.exists():
            print(f"[sync] MISSING source {src}", file=sys.stderr)
            return 1
        text = src.read_text(encoding="utf-8")

        if name in ("arena.js", "arena-styles.css"):
            if "ExtraOrchestra borrowed snapshot" not in text:
                text = NOTE_JS + text
        elif name == "arena.html":
            # HTML: вставим NOTE-коммент сразу после <head> или в начало <html>
            if "ExtraOrchestra borrowed snapshot" not in text:
                note = (
                    f"<!-- ExtraOrchestra borrowed snapshot — webapp/ @ {HEADER_COMMIT}. "
                    f"DO NOT auto-sync; use extra_orchestra/scripts/sync_borrowed.py. -->\n"
                )
                text = note + text

        if name == "arena.js":
            text = _inject_hook(text)

        if dst.exists() and dst.read_text(encoding="utf-8") == text and not args.force:
            print(f"[sync] {name}: up-to-date")
        else:
            dst.write_text(text, encoding="utf-8")
            print(f"[sync] {name}: written ({len(text)} bytes)")

    print(f"[sync] done. snapshot commit = {HEADER_COMMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())