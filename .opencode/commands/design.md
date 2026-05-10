---
description: Opt-in frontend design mode for polished UI/prototype work
subtask: false
---

You are now in opt-in frontend design mode for this request only:

$ARGUMENTS

Use this mode for frontend UI, game screens, visual polish, HTML prototypes, flows, components, layout, typography, motion, and interaction design.

This command is inspired by the public Anthropic `frontend-design` skill and adapted for OpenCode + this ExtraArena project. Do not treat it as a permanent project instruction.

## First Move

Before editing, quickly understand the local design context:

- Inspect only the relevant files in `webapp/`, `web/`, and nearby Python/template files if needed.
- List candidate assets in `DesignAssets/` before opening specific images or audio.
- Prefer existing visual vocabulary, names, states, and interaction patterns over a generic redesign.
- If the request is ambiguous, ask at most one concise clarifying question. If the likely intent is clear, proceed.

## Design Direction

Commit to a clear visual point of view before writing code.

Decide:

- Purpose: what job this screen/component does for the player or user.
- Audience and mood: game UI, tool UI, shop/economy, onboarding, battle, inventory, admin, etc.
- Density: compact operational surface vs expressive game presentation.
- Memorable detail: one specific visual or interaction choice that gives the UI character.

Avoid default AI aesthetics:

- No generic purple/blue gradient hero unless the project already uses it.
- No card-grid-first composition when a denser game/tool layout is more appropriate.
- No decorative blobs/orbs as a substitute for layout.
- Do not default to Inter/Roboto/system-font sameness when a more fitting typography choice exists, but respect project constraints.

## Frontend Quality Bar

Implement working code, not a static sketch, unless the user explicitly asks for a mockup.

Prioritize:

- Clear hierarchy and scan paths.
- Stable dimensions for fixed-format UI: buttons, counters, tiles, boards, toolbars, tabs.
- Responsive behavior across desktop and narrow/mobile widths.
- Text that never overlaps or spills out of controls.
- Intentional color system using existing palette or CSS variables.
- Motion that has a job: feedback, reveal, focus, transition, or game feel. Prefer fewer high-quality moments over scattered animation noise.
- Accessibility basics: labels where useful, visible focus, readable contrast, sane hit targets.

For ExtraArena/game UI:

- Favor practical play readability over marketing-page drama.
- Use assets from `DesignAssets/` selectively when they clarify the actual screen.
- Keep combat/shop/inventory/economy interfaces information-dense but organized.
- Do not hide important controls behind purely decorative composition.

## Implementation Rules

- Match the current stack and file style. For this project, check `webapp/` before adding any new framework or dependency.
- Keep edits scoped to the requested screen/component.
- Prefer CSS variables and existing classes before inventing large parallel styling systems.
- Use icons/assets only when they improve recognition or game feel.
- Do not create a landing page unless the user explicitly asked for one.
- If creating a standalone prototype, place it in `webapp/` with a descriptive filename.
- Do not read or modify `.claude/`, `.venv/`, caches, logs, old worktrees, model binaries, videos, audio, or large asset folders unless explicitly relevant.

## Optional Reference

There is a local reference file:

`.opencode/references/Claude-Design-Sys-Prompt.txt`

Do not read it by default. If the user explicitly asks for "Claude Design-like", "closer to Claude Design", or asks to compare with that prompt, inspect only the relevant sections and adapt principles. Do not copy large chunks into output or project files.

## Verification

After meaningful UI changes:

- Run or open the app/prototype when feasible.
- Check desktop and narrow/mobile viewport behavior.
- Look for text overflow, overlapping controls, blank assets, broken paths, and console/runtime errors.
- If browser verification is not possible, state what was not verified.

## Response Style

- Keep progress updates concise.
- In the final response, mention the files changed and what was visually verified.
- Do not dump internal design theory unless the user asks.
