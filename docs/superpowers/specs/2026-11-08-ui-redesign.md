# UI redesign: "Seven rulebooks, one desk" (2026-11-08)

User feedback after running SP6 locally: the UI is bland and technical. A first-time visitor cannot tell
what the page is or who it is for; the measured numbers mean nothing to non-experts. The Ask page works
and keeps its behaviour. The user gave full creative control: push design, effects and motion as far as
Streamlit allows, and make the app understandable. Name, repo and architecture stay.

## Audience and job

- Audience: a founder, product manager or engineer at a small tech company who has heard "the AI Act"
  or "NIS2" and does not know whether it applies to them. Second: a compliance person checking a policy.
- Primary job of the first screen: say in plain words what this is, who it is for, and give one click
  to a result (a ready-made example on the Checklist, or a question on Ask).

## Concept

The subject is law that people mark up: rulebooks on a desk, coloured index tabs, a highlighter over
the line that applies to you, a stamp on what is forbidden. The memorable element is the **rulebook
stack** on Home: eight index-tabbed rulebooks that fan in once on load. Everything else is quiet.

## Tokens

| Token | Light | Dark | Role |
|---|---|---|---|
| paper | `#F6F7FB` | `#0D1230` | page background (cool, not cream) |
| paper-2 | `#ECEEF6` | `#151B45` | sidebar, quiet panels |
| ink | `#141A47` | `#E7E9F7` | text (deep navy, not black) |
| ultramarine | `#2F45D6` | `#8E9BFF` | primary, links, focus |
| highlighter | `#FFE35A` | `#FFE35A` | "applies to you" marks only |
| stamp | `#C2362F` | `#FF7A70` | prohibitions, declines, errors |
| sage | `#2E7D5B` | `#6FCF9F` | "applies now", grounded |
| amber | `#B86A12` | `#F2A94A` | "coming", waking |
| rule-line | `#D5D9EA` | `#2A3166` | borders |

Each rulebook has its own tab colour (muted, distinct, AA against its label): AI Act ultramarine, GDPR
teal, NIS2 + IR violet, DORA plum, CRA ochre, NIST three in slate shades.

Type: headings **Newsreader** (serif, the voice of the law, opsz 6..72, weights 500/600), body
**Public Sans** (400/500/600, a government sans), code only **JetBrains Mono**. No monospace for small
labels. Sentence case everywhere; no all-caps eyebrows; no "→" glyphs on buttons.

## Layout

```
Home
+---------------------------------------------------------------+
| Which EU and US rules apply to      |   [ AI Act        ]     |
| your AI, data or cyber product?     |    [ GDPR          ]    |
| Plain answers quoted from the        |     [ NIS2         ]   |   <- rulebook stack:
| official text, with the date each    |      [ DORA        ]   |      index tabs fan in
| rule starts to apply.                |       [ CRA        ]   |      once on load;
| [Try a ready-made example] [Ask]     |        [ NIST x3   ]   |      hover lifts one and
+---------------------------------------------------------------+      shows its plain line
| How it works (a real sequence, so numbered)                   |
| 1 Tell it about you  2 See what applies  3 Check your policy  |
+---------------------------------------------------------------+
| The three tools: cards with a one-line "use this when..."     |
+---------------------------------------------------------------+
| Can you trust it? Plain-language proof with a 10x10 dot grid  |
| "Finds the right article in its top 6 results 87 of 100 times"|
| "Refused 13 of 14 questions about rules it does not cover"     |
| link: see the full evaluation                                  |
+---------------------------------------------------------------+
| What is in the library: one row per rulebook, "who it is for" |
+---------------------------------------------------------------+
```

Left aligned throughout. Hero two columns on desktop, stacked on mobile (stack moves under the text).

## Motion

- One orchestrated moment: the rulebook stack fans in on Home load (staggered 60 ms, 500 ms ease-out).
- User-triggered only otherwise: hover lift on a rulebook tab and on preset cards, a highlighter sweep
  when a checklist count changes, button press feedback.
- All motion off under `prefers-reduced-motion: reduce`.

## Plain-language rulebook lines (use exactly; do not invent others)

- EU AI Act: rules for building, selling or using AI in the EU, stricter as the risk rises.
- GDPR: how organisations may collect and use personal data about people in the EU.
- NIS2 and its Implementing Regulation: cybersecurity duties for important sectors such as cloud, digital
  infrastructure, energy and health.
- DORA: digital resilience duties for banks, insurers and other financial firms.
- Cyber Resilience Act: security requirements for hardware and software products sold in the EU.
- NIST SP 800-53, CSF 2.0, AI RMF: voluntary US frameworks of security and AI risk practices.

## Pages

- **Home**: as above. Numbers come from `eval/results` through `services/eval_data.py`; phrase them as
  sentences, keep the exact values, and link to Evaluation for method.
- **Checklist**: presets (`PRESETS` in `services/presenters.py`) as tactile cards at the top with a reset
  action; counts styled as four tabs (applies now / coming / depends / does not apply).
- **Ask**: behaviour unchanged; inherits the theme. Failed turns already render with a retry button.
- **Gap check, Evaluation**: inherit the theme; Evaluation gets a short plain-language intro on top of the
  existing tabs.
- Header: product name as a wordmark left of the nav; API status stays right.

## Quality floor

AA contrast in both themes, visible focus ring (ultramarine, 2 px), works at 375 px wide with no
horizontal scroll, reduced motion respected, no external JS. CSS lives in one file
(`frontend/static/theme.css`, injected once by `services/shell.py`).
