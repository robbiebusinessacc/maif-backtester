# MAIF Testing Framework — website

Public showcase for the MAIF backtester. Deploys to `maif.robbiew.dev`.

## Local dev

```bash
cd site
npm install
npm run dev     # http://localhost:4321
```

## Build

```bash
npm run build   # outputs to dist/
npm run preview # preview the built site locally
```

## Deploy — Cloudflare Pages

1. Push this repo to GitHub.
2. In Cloudflare Pages: *Create Project → Connect to Git → pick the repo*.
3. Build settings:
   - **Framework preset:** Astro
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Root directory:** `site`
   - **Node version:** 18 or 20
4. Custom domain: in the Pages project, *Custom domains → Set up a custom domain → `maif.robbiew.dev`*. Cloudflare will suggest a CNAME — accept it if the zone is already on Cloudflare DNS.

## Structure

```
site/
  public/
    brand/         MAIF logo
    scorecards/    Scorecard PNGs for the 5 built-in strategies (copies from repo root)
    gan/           cGAN regime output PNGs
  src/
    components/    Astro + React islands
    data/          Strategy metadata, representative chart data
    layouts/       Base layout (header + footer)
    pages/         9 pages
    styles/        global.css (tokens, typography, prose)
```

## Brand tokens

See `tailwind.config.mjs` — colors and font families live there. To adjust the palette, edit those tokens; nothing else should need to change.

| Token | Hex | Use |
| --- | --- | --- |
| `primary` | `#881C1C` | Minuteman Maroon — links, pills, eyebrow labels |
| `primary-dark` | `#5F1111` | Hover/pressed |
| `ink` | `#0F0F0F` | Body text |
| `paper` | `#FAF7F2` | Background (warm off-white) |
| `muted` | `#8B8680` | Secondary text, axis labels |
| `hair` | `#E8E4DE` | Hairline rules, borders |
| `positive` | `#2E7D32` | Positive returns / A grades |
| `negative` | `#C62828` | Negative returns / F grades |
| `aux` | `#7B6D5C` | Secondary chart series |

Fonts: `Cormorant Garamond` (display), `Inter` (body), `JetBrains Mono` (code / tickers).

## Content pipeline (future work)

The interactive equity-race chart on the landing page uses representative data in `src/data/equity_race.ts`. To swap in real per-day equity series, add `export_scorecard_json()` to `backtester/scorecard.py`, regenerate scorecards for the five built-ins, and replace the hardcoded `values` arrays.

Scorecard detail pages (`/scorecards/[id]`) currently render the PNG pages only. Once JSON data is exported, add interactive chart components that read from `/public/scorecards/{id}.json` and replace the appropriate PNG sections.

## TODOs to finish

Search for `TODO` in the codebase. The ones that matter most:

- Real GitHub repo URL (footer, get-started page)
- Real Colab notebook URL (get-started page, landing CTA)
- Pull exact return / Sharpe / grade figures for Turtle, Bracket Breakout, and BTC SMA from the scorecard PNGs into `src/data/strategies.ts` — currently marked `null`.
