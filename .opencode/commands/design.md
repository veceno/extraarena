---
description: Optional design workflow for UI/prototype work
subtask: false
---

Enter design mode for this request: $ARGUMENTS

Use this only for UI, visual design, HTML prototypes, frontend polish, screens, flows, or game-facing presentation work.

Work like a senior product designer and frontend engineer:

- First inspect the relevant existing UI files and nearby assets with targeted searches.
- Prefer the project's current visual language in `webapp/`, `web/`, and `DesignAssets/`.
- Build the actual usable screen or prototype first, not a generic landing page.
- Keep layouts responsive, readable, and stable across desktop and mobile widths.
- Use assets selectively. Do not bulk-read or copy large asset folders.
- Keep generated code maintainable and scoped to the requested artifact.
- Verify visually when possible, and say what was not verified if a browser check is not possible.

Cost discipline:

- Do not read `.opencode/references/Claude-Design-Sys-Prompt.txt` by default.
- If the user explicitly asks to emulate Claude Design more closely, inspect only the relevant sections of that reference file and summarize the applicable principles instead of copying it into output.
- Avoid loading videos, audio, model binaries, virtualenvs, caches, logs, or old worktrees unless the user asks.
