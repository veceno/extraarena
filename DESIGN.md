---
version: alpha
name: ExtraArena
primary: "#f5921e"
description: |
  Telegram Mini App card-battle game. Two visual modes share a
  purple-dark foundation: the organic glow-driven main app and
  the flat bold-block arena interface.
colors:
  # ── React SPA (main app) palette ──
  primary: "#f5921e"
  canvas: "#0f0a1a"
  canvas-deep: "#130d26"
  surface: "#1a1030"
  surface-elevated: "#2d1f52"
  border: "#4a3d6a"
  border-light: "#7a6fa0"
  text-primary: "#f0ecff"
  text-secondary: "#c4b8e8"
  text-muted: "#7a6fa0"
  orange: "#f5921e"
  orange-dark: "#d97510"
  orange-light: "#ffb347"
  gold: "#fbbf24"
  gold-bright: "#ffd700"
  red: "#ef4444"
  red-soft: "#f87171"
  pink: "#e040c0"
  pink-bright: "#f472b6"
  teal: "#2dd4bf"
  teal-dark: "#14b8a6"
  green: "#4ade80"
  green-soft: "#51cf66"
  green-lime: "#a3e635"
  blue: "#60a5fa"
  purple-light: "#c084fc"
  purple-highlight: "#7c5cbf"
  purple-soft: "#a78bfa"
  purple-pale: "#9c7de0"
  purple-mid: "#5b3fa0"
  purple-deep: "#3d2a70"

  # ── Arena battle palette ──
  arena-bg: "#3E0B53"
  arena-avatar: "#9820CC"
  arena-hp: "#ED230D"
  arena-name: "#73189A"
  arena-mana-bg: "#2D0A40"
  arena-mana-fill-start: "#FF9400"
  arena-mana-fill-end: "#FF6B00"
  arena-cta: "#FF9400"
  arena-card-bg-start: "#4A148C"
  arena-card-bg-end: "#6A1B9A"
  arena-card-selected: "#FFD700"
  arena-turn-player: "#4ADE80"
  arena-turn-opponent: "#FB7185"
  arena-target-glow-attack: "#FF3B48"
  arena-target-glow-heal: "#4ADE80"
  arena-freeze: "#00bfff"
  arena-damage-particle: "#FF4D00"
  arena-stat-attack: "#FF6B6B"
  arena-stat-health: "#51CF66"

  # ── Rarity (Collection) ──
  rarity-common: "#94a3b8"
  rarity-rare: "#60a5fa"
  rarity-superrare: "#2dd4bf"
  rarity-epic: "#a78bfa"
  rarity-legendary: "#fbbf24"
  rarity-mythic: "#f472b6"
  rarity-divine: "#e0e7ff"
  rarity-limited: "#f87171"
  rarity-unique: "#c084fc"
  rarity-start: "#94a3b8"

  # ── Rarity (Case Open) ──
  case-rarity-common: "#9ca3af"
  case-rarity-rare: "#60a5fa"
  case-rarity-superrare: "#a78bfa"
  case-rarity-epic: "#f59e0b"
  case-rarity-legendary: "#f97316"
  case-rarity-mythic: "#ec4899"
  case-rarity-divine: "#fcd34d"
  case-rarity-limited: "#f43f5e"

  # ── Tier colors (case rolling) ──
  tier-2: "#6ee7b7"
  tier-3: "#60a5fa"
  tier-4: "#a78bfa"
  tier-5: "#f59e0b"
  tier-6: "#f43f5e"

  # ── League (Glory Path) ──
  league-novice: "#2ECC71"
  league-bronze: "#E67E22"
  league-silver: "#95A5A6"
  league-gold: "#F1C40F"
  league-crystal: "#3498DB"
  league-master: "#F39C12"
  league-champion: "#E74C3C"
  league-grandmaster: "#9B59B6"
  league-legendary: "#FF6B6B"
  league-extra: "#FFD700"

  # ── Shop item image tiers ──
  shop-common: "#9ca3af"
  shop-rare: "#3b82f6"
  shop-epic: "#9333ea"
  shop-legendary: "#fbbf24"
  shop-mythic: "#ef4444"
  shop-divine: "#fbbf24"

  # ── Chibi / legacy CSS theme ──
  chibi-bg: "#1a0f2e"
  chibi-card: "#2d1b4e"
  chibi-border: "#4a2c6b"
  chibi-text: "#ffffff"
  chibi-text-muted: "#d1d5db"
  chibi-pink: "#ff9ec8"
  chibi-purple: "#c084fc"
  chibi-blue: "#60a5fa"
  chibi-gold: "#fbbf24"
  chibi-red: "#ef4444"
  chibi-orange: "#f97316"
  chibi-extrapass: "#f59e0b"

typography:
  headline:
    fontFamily: "Exo 2, sans-serif"
    fontSize: 24px
    fontWeight: "700"
    lineHeight: 32px
  title:
    fontFamily: "Exo 2, sans-serif"
    fontSize: 20px
    fontWeight: "600"
    lineHeight: 28px
  body:
    fontFamily: "Inter, sans-serif"
    fontSize: 16px
    fontWeight: "400"
    lineHeight: 24px
  body-sm:
    fontFamily: "Inter, sans-serif"
    fontSize: 14px
    fontWeight: "400"
    lineHeight: 20px
  label:
    fontFamily: "Inter, sans-serif"
    fontSize: 12px
    fontWeight: "600"
    lineHeight: 16px
    letterSpacing: 0.02em
  caption:
    fontFamily: "Inter, sans-serif"
    fontSize: 10px
    fontWeight: "500"
    lineHeight: 14px
  arena-headline:
    fontFamily: "Futura PT, Arial Black, sans-serif"
    fontSize: 28px
    fontWeight: "900"
    lineHeight: "1"
  arena-body:
    fontFamily: "Futura PT, Arial Black, sans-serif"
    fontSize: 14px
    fontWeight: "700"
    lineHeight: "1"
  chibi-body:
    fontFamily: "FuturaPT, Comic Sans MS, Comic Sans, Arial Rounded MT Bold, sans-serif"
    fontSize: 16px
    fontWeight: "400"
    lineHeight: 24px
  chibi-heading:
    fontFamily: "FuturaPT, Comic Sans MS, Comic Sans, Arial Rounded MT Bold, sans-serif"
    fontSize: 20px
    fontWeight: "700"
    lineHeight: 28px

rounded:
  none: 0px
  xs: 4px
  sm: 7px
  md: 10px
  lg: 14px
  xl: 16px
  "2xl": 20px
  "3xl": 24px
  "4xl": 30px
  full: 9999px

spacing:
  unit: 8px
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 20px
  "2xl": 24px
  "3xl": 32px

components:
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.xl}"
    padding: "{spacing.lg}"
  card-elevated:
    backgroundColor: "{colors.surface-elevated}"
    rounded: "{rounded.xl}"
    padding: "{spacing.lg}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.text-primary}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: 12px 24px
  button-primary-hover:
    backgroundColor: "{colors.orange-light}"
  button-secondary:
    backgroundColor: "{colors.surface-elevated}"
    textColor: "{colors.text-primary}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: 12px 24px
  button-secondary-hover:
    backgroundColor: "{colors.purple-deep}"
  button-danger:
    backgroundColor: "{colors.red}"
    textColor: "{colors.text-primary}"
    typography: "{typography.label}"
    rounded: "{rounded.lg}"
    padding: 12px 24px
  toggle:
    backgroundColor: "{colors.surface-elevated}"
    rounded: 13px
    height: 25px
    width: 44px
  toggle-active:
    backgroundColor: "{colors.purple-pale}"
  input-field:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: 12px 16px
  badge:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.text-primary}"
    typography: "{typography.caption}"
    rounded: "{rounded.full}"
    padding: 2px 8px
  arena-card:
    backgroundColor: "#4A148C"
    rounded: "{rounded.sm}"
    padding: 0px
  arena-cta:
    backgroundColor: "{colors.arena-cta}"
    textColor: "#FFFFFF"
    typography: "{typography.arena-headline}"
    rounded: "{rounded.sm}"
    padding: 12px 24px
  arena-mana-bar:
    backgroundColor: "rgba(0,0,0,0.4)"
    rounded: "{rounded.sm}"
    height: 26px
  arena-tooltip:
    backgroundColor: "rgba(20,10,35,0.98)"
    rounded: "{rounded.md}"
    padding: 10px 14px
---

## Overview

ExtraArena is a Telegram Mini App that combines card collecting, looter-shooter case
opening, and real-time PvP arena battles. The visual identity is built on a **deep
purple-dark foundation** with two distinct visual modes:

**The Main App** — an organic, glow-infused UI with layered transparency, subtle
backdrop blur, and pulsating ambient light. Purple gradients dominate, offset by
a warm orange accent that drives every primary action. Typography pairs the
geometric clarity of **Exo 2** for headlines with the clean readability of **Inter**
for body text. Cards and panels float with soft drop shadows and subtle purple glows,
creating a sense of depth without heaviness.

**The Arena Battle Interface** — a complete tonal shift into a flat, bold, block-based
layout. Every element is a solid-color "island" — no shadows, no transparency, no
blur. Deep magenta-purple (`#3E0B53`) fills the background while the UI is
constructed from four opaque blocks: Avatar (`#9820CC`), HP (`#ED230D`), Name
(`#73189A`), and Mana/CTA (`#FF9400`). Typography is exclusively **Futura PT**
at weight 900 (ExtraBold), optimized for split-second readability in combat.
The aesthetic is confrontational, urgent, and designed for mobile portrait.

The two modes share a common purple bloodline but diverge in philosophy: the main
app invites exploration through atmosphere; the arena demands action through clarity.

## Colors

### Main App Palette

The core palette layers five shades of deep purple from near-black canvas to pale
accent, forming a monochromatic depth stack:

- **canvas** (`#0f0a1a`) — The root background. Nearly black with a hint of violet.
  Never used as a surface; it's the void behind the UI.
- **canvas-deep** (`#130d26`) — Used for inset panels and darker recessed areas.
- **surface** (`#1a1030`) — Standard card and panel background. The workhorse
  mid-tone.
- **surface-elevated** (`#2d1f52`) — Interactive card hover states, modal headers,
  and highlighted containers.
- **border** (`#4a3d6a`) — Subtle dividers and outlines.
- **border-light** (`#7a6fa0`) — More visible borders when needed.

**Text** follows a three-step hierarchy:
- **text-primary** (`#f0ecff`) — All primary body and heading text. An off-white
  with a hint of lavender that feels warmer than pure white.
- **text-secondary** (`#c4b8e8`) — Supporting text, descriptions, metadata.
- **text-muted** (`#7a6fa0`) — Placeholder text, timestamps, tertiary information.

**Orange** (`#f5921e`) is the sole action color and the most prominent accent.
It is the `primary` token, used for CTA buttons, active states, selected indicators,
energy/mana visualizations, and ExtraPass branding. Its dark variant (`#d97510`)
serves pressed states; its light variant (`#ffb347`) serves hover highlights.

**Gold** (`#fbbf24`) signals value and reward — legendary rarity indicators,
shop highlights, coin/gem accents. A brighter variant (`#ffd700`) is used for
selected card borders in the arena.

**Red** (`#ef4444`) is reserved for destructive actions, HP damage indicators
in the main app, and error states.

**Teal** (`#2dd4bf`) identifies AI opponents and AI-themed elements. Its dark
variant (`#14b8a6`) provides the pressed state. AI panels use a subtle teal tint
achieved through low-opacity overlays (`#2dd4bf` at 6% for backgrounds, 18% for
borders) over the dark purple canvas.

**Pink** (`#e040c0`) represents the Midoria AI opponent model. A second pink
variant (`#f472b6`) is used for mythic rarity indicators. Midoria panels use
the same low-opacity overlay technique (`#e040c0` at 6% and 18%).

**Green** (`#4ade80`) marks positive states: victory indicators, heal actions,
"available to attack" glows, and the Lite difficulty level. A darker variant
(`#51cf66`) labels health stats.

**Blue** (`#60a5fa`) appears for rare-tier items, informational elements, and the
Crystal league.

### Arena Battle Palette

The arena discards all transparency and glow in favor of a solid-color zoning
system. Every UI island has exactly one background color and no border. When an
element needs to indicate a state change (e.g. ExtraPass active), a linear gradient
is applied within the same purple family rather than introducing borders or shadows.

The arena card selection and hover glow is achieved with a `#FFD700` border and a
`0 0 30px rgba(255,215,0,0.9)` box-shadow — the only glow effect tolerated in
the arena.

The mana bar fill uses a linear gradient from `#FF9400` to `#FF6B00`, animating
width changes with a 400ms cubic-bezier ease.

### Rarity, Tier & League Colors

The product uses **three separate rarity color maps** — one for the Collection
screen, one for Case Opening, and one for the Shop item borders. These maps
share the same rarity names but shift hues to match each context's energy.

**Leagues** (Glory Path) form a visual progression from novice green through
bronze, silver, gold, crystal blue, master amber, champion red, grandmaster
purple, legendary pink-red, and finally Extra gold (`#FFD700`). Each league
color is used for badges, borders, and milestone markers.

**Case tiers** increase from T2 through T6, color-shifting from emerald green
through sapphire blue, soft purple, amber, and finally deep rose — lower tiers
remain cool and approachable while higher tiers feel rare and dangerous.

## Typography

### Main App

- **Headlines** use **Exo 2** at weight 700. Its geometric construction and
  wide proportions give headers a futuristic, sport-like presence that contrasts
  with the organic background. The 900 weight (Black) is reserved for numeric
  displays and league names.
- **Body text** uses **Inter** at weights 400–600. Its neutral, highly legible
  character ensures long card names, stats, and descriptions remain readable at
  small sizes on mobile screens.
- **Labels and captions** use Inter at weights 600 and 500 respectively, with
  tight letter-spacing for button text and metadata.

### Arena

The arena uses **Futura PT** exclusively, loaded at weights 700 (Bold) and 900
(ExtraBold). This typeface was chosen for its sharp, geometric, almost industrial
character — it reads like a sports scoreboard. All text in the arena is set in
ExtraBold (900) with `line-height: 1` and zero text-shadow, except for mana
costs on cards which use a black text-shadow stroke effect for contrast against
the card art.

### Chibi / Legacy

The legacy CSS-based theme uses **FuturaPT** (the Book weight family) with a
fallback stack that includes Comic Sans MS and Arial Rounded MT Bold — giving
it a playful, chibi-styled personality distinct from the modern React app.

## Layout & Spacing

The product is designed **mobile-first** for Telegram's WebView with a single-column
layout. The main app uses an 8px base spacing grid with generous internal padding and a
`max-width` of 480px centered on wider viewports to prevent stretching on tablets.

The arena uses a **grid-based zone layout** optimized for portrait orientation:
- Top zone (opponent info + board slots) — 35% height
- Middle zone (turn indicator) — compact bar
- Bottom zone (player hand + info) — 65% height

Within the arena, the player panel uses a `2fr / 1fr` grid — the left 66% holds
avatar, HP, name, and mana in a stacked column; the right 33% holds the end-turn
CTA button at full height.

Element gaps follow an 8px or 12px rhythm in the arena, creating tight visual
groupings that read as unified "islands" of information.

## Elevation & Depth

### Main App

Depth is created through three mechanisms:

1. **Background gradient layers** — radially animated purple/pink/blue orbs with
   `blur(60px)` create ambient lighting behind the content. The loading screen
   uses a `#0a0514` base with three overlaid radial gradient layers blending
   `#c084fc`, `#60a5fa`, and `#ff9ec8` at low opacities.
2. **Surface stacking** — `canvas` → `canvas-deep` → `surface` → `surface-elevated`
   provides four tonal layers. Each card uses `box-shadow: 0 4px 16px rgba(0,0,0,0.3)`;
   elevated panels use `0 8px 32px rgba(0,0,0,0.5)`. Purple glow shadows
   (`0 0 20px rgba(192,132,252,0.4)`) backlight interactive cards from behind.
3. **Backdrop blur** — Modal overlays use `backdrop-filter: blur(8px)` over a
   `rgba(0,0,0,0.7)` backdrop. The loading screen uses `backdrop-filter:
   blur(20px) saturate(180%)` over a `rgba(26,15,46,0.85)` to `rgba(45,27,78,0.9)`
   gradient, creating a premium crystalline glass effect.

AI opponent panels and Midoria panels achieve depth through low-opacity colored
overlays rather than elevation — `rgba(45,212,191,0.06)` for AI and
`rgba(224,64,192,0.06)` for Midoria backgrounds, with borders at 0.18 opacity.

### Arena

The arena has **zero elevation**. No box-shadow, no backdrop-filter, no
transparency. Every element sits flat on a single visual plane at `#3E0B53`.
The only "depth" cues are:

- A `rgba(0,0,0,0.3)` overlay on drag-and-drop slots
- The gold border glow (`0 0 30px rgba(255,215,0,0.9)`) on selected or hovered
  cards
- Pulsating green (`rgba(76,217,100,0.5)`) and red (`rgba(255,59,60,0.5)`)
  glows indicating which units can attack or are targetable
- Orange tooltip border (`rgba(255,148,0,0.5)`) on `rgba(20,10,35,0.98)` card
  detail popups

This flatness is deliberate — it removes visual ambiguity during fast-paced
combat where every millisecond of readability counts.

## Shapes

The main app leans **soft and rounded**. Cards use `16px` corner radii. Buttons
use `14px`. Toggle switches, mana circles, and badges use `50%` or `9999px` for
fully round pill shapes. The loading screen content box uses a generous `30px`
radius. Backdrop-filter blurred containers and glow shadows further soften the
silhouettes.

The arena inverts this philosophy with **hard-edged, squared blocks**. Every
info block, mana bar, card, and button uses a `7px` border radius — enough to
avoid sharp corners but tight enough to read as structural and architectural.
This aligns with the scoreboard aesthetic and keeps pixel measurements precise
in the grid layout.

## Components

### Main Application Components

**Cards** are the fundamental container. Standard cards sit on `surface`
(`#1a1030`) with `rounded.xl` (16px) and a soft `0 4px 16px rgba(0,0,0,0.3)`
drop shadow. Elevated cards use `surface-elevated` (`#2d1f52`) for interactive
states and modal headers. Gradient cards incorporate the purple highlight in a
linear gradient for promotional or premium content. A glowing card variant adds
`0 0 20px rgba(192,132,252,0.4)` for special or highlighted cards.

**Buttons** follow a three-tier scheme:
- **Primary** — solid orange (`#f5921e`) on all CTAs and major actions.
  Hover/press shifts to orange-light (`#ffb347`).
- **Secondary** — purple background on `surface-elevated` (`#2d1f52`), used for
  navigation and less critical actions. Hover shifts to purple-deep (`#3d2a70`).
- **Danger** — red (`#ef4444`) for destructive operations like surrendering or
  deleting.

All buttons use `rounded.lg` (14px), uppercase labels, and a 150ms–200ms ease
transition on hover/press.

**Toggles** are 44×25px pills with a 13px radius. The inactive state uses
`surface-elevated` (`#2d1f52`); the active state uses `purple-pale` (`#9c7de0`)
with a white knob and soft purple glow.

**Badges** are small circular pills using orange fill with white text at 10px,
used for notification counts, ExtraPass indicators, and status markers.

**Input fields** use the standard surface color for backgrounds with padding of
12px vertical and 16px horizontal, and a 14px border radius.

### Arena Components

**Info Islands** are the core arena building blocks. Each island is a
solid-color `div` with `rounded.sm` (7px), no border, and no shadow. The
avatar (`#9820CC`), HP (`#ED230D`), name (`#73189A`), and mana (`#2D0A40`)
blocks stack vertically in the left column while the CTA button fills the
right column.

**Arena Cards** are 7px-radius panels with a `linear-gradient(135deg, #4A148C,
#6A1B9A)` background. Attack stat text is `#FF6B6B`; health stat text is
`#51CF66`. A golden border (`#FFD700`) with a `0 0 30px rgba(255,215,0,0.9)`
glow appears on selection (`translateY(-12px)`) or hover. Cards with
insufficient mana or disabled status are dimmed to 40-50% opacity and may
receive a grayscale filter.

**Mana Bar** uses a dark transparent track (`rgba(0,0,0,0.4)`) with 7px radius
and a `linear-gradient(90deg, #FF9400, #FF6B00)` fill that animates width
changes with a 400ms cubic-bezier ease.

**Turn Indicator** is a centered plaque using `rgba(0,0,0,0.5)` with a green
tint (`rgba(74,222,128,0.2)`) during the player's turn and a red tint
(`rgba(251,113,133,0.2)`) during the opponent's turn. The turn text itself
shifts to `#4ADE80` or `#FB7185` accordingly.

**Arena Tooltips** use a near-opaque dark purple background
(`rgba(20,10,35,0.98)`) with an orange border (`rgba(255,148,0,0.5)`) for
card detail popups during combat.

### Status Effects (Arena)

Card status effects are communicated through CSS filters and dedicated icon
overlays:
- **Shield** — `grayscale(1) brightness(0.8)` on the card art
- **Frozen** — `hue-rotate(180deg) brightness(1.2) saturate(0.5)` with a
  `#00bfff` counter badge and blue drop-shadow on the freeze icon
- **Sleep** — `grayscale(0.5) contrast(0.8)` on the card art
- **Charge** — `#FCD34D` icon color
- **Heal target** — pulsating `#4ADE80` glow icon on friendly units
- **Attack target** — pulsating red glow icon on enemy units
- **Deathrattle** — `#A855F7` border

### Potion & Extrapass Visuals

Potions display with a `#7000ff` to `#00ff88` gradient background and
`#A855F7` border. ExtraPass-active elements receive a premium
`#73189A → #9820CC → #FF9400` gradient with an orange border and glow
(`0 0 15px rgba(255,148,0,0.3)`).

## Do's and Don'ts

- **Do** use the orange-primary for exactly one CTA per screen — it is the
  dominant action color and should never compete with itself.
- **Do** use `surface` and `surface-elevated` strictly in that order; never
  place a surface-elevated card beneath a surface card visually.
- **Do** maintain the arena's flat aesthetic — no shadows, no backdrop blur,
  no gradients beyond the card background and mana bar fill.
- **Don't** mix the arena and main-app color tokens within the same interface.
- **Don't** use gold (`#fbbf24` / `#ffd700`) for primary actions — gold is
  reserved for reward/value indicators and card selection.
- **Don't** apply rounded radii larger than `7px` in the arena — the
  scoreboard aesthetic depends on tight, geometric corners.
- **Don't** exceed weight 700 for body text in the main app — the heavier
  weights belong to Exo 2 headlines only.
