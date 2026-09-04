# FLTest brand kit

The mark is a **test matrix**. Nine cells: two solid on the rising diagonal, one lifted clear of
the grid in orange. That's the configuration under test — the thing FLTest isolates out of the
attack × defense × framework sweep. Empty cells sit at 45% opacity so they recede; the solid tiles
carry a soft shadow so they read as sitting above the grid.

Open `preview.html` to see every asset rendered, including 16 px and on-maroon versions.

---

## Colors

Built on the Virginia Tech palette. Maroon carries the brand, orange is an accent only, and the
neutrals are warmed toward Hokie Stone so nothing looks blue next to the maroon.

### Maroon (primary) — `#861F41` is VT Chicago Maroon

| Step | Hex | Use |
| --- | --- | --- |
| 50 | `#FBF3F6` | tinted page sections |
| 100 | `#F5E2E8` | table stripes, hover fills |
| 200 | `#E9C4D0` | soft borders |
| 300 | `#D598AC` | disabled text on maroon |
| 400 | `#B3607F` | hover state on dark |
| **500** | **`#861F41`** | **primary: header, links, mark** |
| 600 | `#741A38` | pressed state |
| 700 | `#5E142D` | dark scheme primary |
| 800 | `#480F23` | dark surface raised |
| 900 | `#310A18` | dark page background, code blocks |

### Orange (accent) — `#E5751F` is VT Burnt Orange

| Step | Hex | Use |
| --- | --- | --- |
| 50 | `#FEF6EE` | callout background |
| 100 | `#FDE8D3` | callout border |
| 200 | `#FACCA3` | chart fills |
| 300 | `#F5AC6C` | **accent on maroon backgrounds** (4.81:1) |
| 400 | `#EE8F41` | hover on dark |
| **500** | **`#E5751F`** | **accent: the lifted tile, rules, badges** |
| 600 | `#C55F13` | accent hover on light |
| 700 | `#9D4A0E` | **orange body text on white** (6.14:1) |
| 800 | `#76370A` | — |
| 900 | `#4F2406` | — |

### Stone (neutral) — `#75787B` is VT Hokie Stone

`50 #FAF8F8` · `100 #F2EEEF` · `200 #E3DCDE` · `300 #C8BFC2` · `400 #9B9295` ·
`500 #75787B` · `600 #5B5457` · `700 #433D3F` · `800 #2B1B20` · `900 #1A1013`

Body text is `stone.800`, muted text `stone.600`, borders `stone.200`.

### Semantic — test outcomes

| Token | Hex | Meaning |
| --- | --- | --- |
| `pass` | `#1F7A4C` | test passed, defense held |
| `fail` | `#B3261E` | test failed, attack succeeded |
| `warn` | `#E5751F` | pitfall detected |
| `info` | `#3B6EA5` | neutral run information |

`fail` is a warmer, lighter red than the maroon on purpose, so a failure badge never reads as
chrome. Never use maroon itself as an error color.

### Contrast, measured

| Pair | Ratio | Verdict |
| --- | --- | --- |
| maroon.500 on white | 9.19 | anything |
| white on maroon.500 | 9.19 | anything |
| stone.800 on white | 16.40 | anything |
| stone.600 on white | 7.37 | anything |
| orange.700 on white | 6.14 | body text ✅ |
| orange.300 on maroon.500 | 4.81 | body text ✅ |
| orange.500 on white | 3.05 | 24 px+ or bold 19 px+ only |
| orange.500 on maroon.500 | 3.02 | large display only |

The one trap: `#E5751F` on white fails for body text. Use `orange.700` for orange prose and links
on light backgrounds, and keep `orange.500` for shapes, rules, and headline-scale type.

---

## Typography

- Display and wordmark: **Space Grotesk** SemiBold.
- Body: **Geist**, falling back to Inter and the system stack.
- Mono: **JetBrains Mono**.
- The wordmark is Space Grotesk SemiBold at −1.5% tracking in single-ink maroon, converted to
  outlines. No font file is needed to render the logo.

Single ink, not two-tone: the mark already spends the orange on the lifted tile, and repeating it
in the word makes the accent stop meaning anything. `fltest-lockup-two-tone.svg` is there if you
disagree.

---

## The shadow

The lift is an `feDropShadow` filter — `0 14px 16px rgba(61,10,28,0.30)`, exposed in CSS as
`--fltest-lift` so UI elements can match. Two things to know:

1. Some SVG-to-PNG converters silently drop filters. If a raster export comes out flat, that's the
   converter, not the file. `rsvg-convert` and any browser handle it correctly; CairoSVG does not.
2. Shadows turn to mud below about 24 px and vanish in one-color printing. Use
   `fltest-mark-flat.svg` there — same geometry, no filter.

---

## Files

```
svg/
  fltest-mark.svg                    primary icon, light backgrounds
  fltest-mark-flat.svg               same, no shadow — print and small sizes
  fltest-mark-on-maroon.svg          icon for #861F41 backgrounds
  fltest-mark-mono-maroon.svg        single-ink maroon
  fltest-mark-mono-white.svg         single-ink white (reversed)
  fltest-favicon.svg                 simplified: grid dropped, three tiles enlarged
  fltest-favicon-on-maroon.svg
  fltest-tile-maroon.svg             rounded maroon tile, for avatars and app icons
  fltest-tile-maroon-square.svg      hard-cornered, for platforms that mask
  fltest-lockup.svg                  icon + wordmark, horizontal
  fltest-lockup-on-maroon.svg
  fltest-lockup-two-tone.svg         orange "Test", if you want it
  fltest-lockup-mono-maroon.svg
  fltest-lockup-mono-white.svg
  fltest-lockup-stacked.svg          icon above wordmark
  fltest-lockup-stacked-on-maroon.svg
  fltest-wordmark*.svg               wordmark alone, four colorways
  fltest-social-preview.svg          1280×640 OpenGraph card
png/                                 raster exports, 16 → 1024 px, plus apple-touch-icon
favicon.ico                          16/32/48/64 bundled
css/brand.css                        CSS custom properties + MkDocs Material bindings
tokens/fltest-tokens.json            same values as JSON, for Tailwind or any pipeline
mkdocs-snippet.yml                   config to paste into the docs site
```

---

## Rules

- **Clear space** on every side is at least one grid cell — about 25% of the icon's width.
- **Minimum sizes.** Full mark: 32 px. Below that use `fltest-favicon.svg`, which drops the empty
  cells and enlarges the three solid tiles into a staircase. Lockup: 140 px wide; below that,
  icon only.
- **Don't** recolor outside the supplied colorways, rotate the mark, change the diagonal's
  direction, fill the empty cells, or set the wordmark in another typeface.
- **Orange stays an accent.** One tile in the mark, and nothing else in the same composition.
- On photos or busy backgrounds, use `fltest-tile-maroon.svg` or the mono-white mark.
