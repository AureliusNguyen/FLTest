# FLTest brand kit — read me first

Same folder layout as before, so this drops in as a straight replacement: `css/`, `png/`, `svg/`,
`tokens/`, `favicon.ico`, `preview.html`.

- `BRAND.md` — color scales, contrast measurements, and usage rules. Read before touching colors.
- `preview.html` — open in a browser to see every asset rendered.
- `mkdocs-snippet.yml` — config for the docs site.

## Install into the docs site

From the root of the `SEED-VT/FLTest` repo:

```bash
mkdir -p docs/assets/logos docs/stylesheets
cp fltest-brand-v2/svg/*.svg  docs/assets/logos/
cp fltest-brand-v2/png/fltest-social-preview-1280x640.png docs/assets/logos/fltest-social-preview.png
cp fltest-brand-v2/png/apple-touch-icon.png docs/assets/logos/
cp fltest-brand-v2/css/brand.css docs/stylesheets/brand.css
```

Then merge `mkdocs-snippet.yml` into `mkdocs.yml`. The key lines are `theme.logo`,
`theme.favicon`, `palette.primary: custom`, `palette.accent: custom`, and `extra_css`. Material
only honors the CSS variables when the palette is set to `custom`.

For the GitHub repo: Settings → Social preview → upload
`png/fltest-social-preview-1280x640.png`. For the org avatar, use `png/fltest-tile-512.png`.

For the repo README, the lockup that works in both GitHub themes:

```markdown
<p align="center">
  <img src="docs/assets/logos/fltest-lockup.svg#gh-light-mode-only" width="420" alt="FLTest">
  <img src="docs/assets/logos/fltest-lockup-mono-white.svg#gh-dark-mode-only" width="420" alt="FLTest">
</p>
```

## Using the colors in code

```css
.pitfall-callout {
  border-left: 4px solid var(--fltest-accent);
  background: var(--fltest-orange-50);
  color: var(--fltest-text);
}

.result-card { box-shadow: var(--fltest-lift); }   /* matches the mark's lift */
```

Same values in `tokens/fltest-tokens.json` for Tailwind, a plotting theme, or anything else. For
matplotlib figures, maroon.500 → orange.500 → stone.500 → maroon.300 → orange.300 gives five
distinguishable series that still read as the brand.

## Two things that will bite you

1. **Filters.** The mark's lift is an `feDropShadow`. CairoSVG and some other converters silently
   ignore filters, so a raster export can come out flat. Use `rsvg-convert` or a browser. All PNGs
   here were rendered with rsvg and are correct.
2. **Orange on white.** `#E5751F` is 3.05:1 — it fails for body text. Use `#9D4A0E`
   (`orange.700`) for orange prose and links on light backgrounds.

## Prompt for Claude Code

> Wire the FLTest brand kit into this repo. The kit is in `fltest-brand-v2/`. Read
> `fltest-brand-v2/BRAND.md` first and follow its rules exactly — especially the contrast table:
> `#E5751F` fails on white for body text, so use `orange.700` (`#9D4A0E`) for orange prose and
> links on light backgrounds, and `orange.300` (`#F5AC6C`) for accents on maroon.
>
> 1. Copy `svg/` to `docs/assets/logos/`, `css/brand.css` to `docs/stylesheets/brand.css`, and the
>    social preview PNG to `docs/assets/logos/fltest-social-preview.png`.
> 2. Merge `mkdocs-snippet.yml` into `mkdocs.yml` without dropping the existing nav, plugins, or
>    markdown_extensions.
> 3. Add the light/dark logo block from `README.md` to the top of the repo README.
> 4. Run `mkdocs build --strict` and fix anything that breaks.
>
> Don't invent new hex values. Every color must come from `tokens/fltest-tokens.json`.

## Changing the wordmark typeface

The wordmark is Space Grotesk SemiBold, outlined. Swapping it means regenerating the lockups, the
stacked lockup, the wordmark files, and the social card — ask rather than editing paths by hand.
