# Design - Emotion Detector

A locked design system for this app. Every page redesign reads this file before
emitting code. Do not regenerate per page - extend or amend this file when the
system needs to grow.

## Genre

editorial (Newsprint register: light warm paper, print metaphor, static)

## Macrostructure family

- Marketing pages: none in this product.
- App pages: Specimen - numbered section labels (`01 / Input`), serif display,
  hairline rules, stacked section heads (label above heading, always one
  column). Variation knob: content grid (single column vs 7/3 form + margin note).
- Content pages: same as app pages (ledger variant: tabular rows, hairline
  separators, mono numerals).

## Theme

Newsprint (light paper band, warm anchor hue ~85, signal-red accent).

- `--color-paper`   oklch(97% 0.010 85)
- `--color-paper-2` oklch(94.5% 0.012 85)
- `--color-paper-3` oklch(91% 0.014 84)
- `--color-rule`    oklch(84% 0.014 82)
- `--color-rule-2`  oklch(58% 0.016 80)
- `--color-muted`   oklch(44% 0.014 75)
- `--color-ink-2`   oklch(30% 0.014 75)
- `--color-ink`     oklch(17% 0.012 70)
- `--color-accent`  oklch(52% 0.170 27)
- `--color-accent-ink` oklch(97% 0.010 85)
- `--color-focus`   oklch(52% 0.170 27)

Emotion colours are data, not decoration. They live as named tokens
(`--emo-joy`, `--emo-anger`, `--emo-fear`, `--emo-sadness`, `--emo-surprise`,
`--emo-disgust`, `--emo-neutral`), deepened from the training palette so they
read on light paper. Server context keys (`color`, `predicted_color`) stay
untouched for tests and JSON consumers; templates reference the tokens.

## Typography

- Display: Fraunces, weight 700, style normal (italic `em` inside display heads
  allowed, never whole headings italic)
- Body: IBM Plex Sans, weight 400 (500 for labels and buttons)
- Mono (outlier, exactly two slots): IBM Plex Mono - ledger numerals and the
  footer colophon. Masthead issue line uses the display face in small caps.
- Display tracking: -0.02em; label tracking: 0.10em uppercase
- Type scale anchor: `--text-display` = clamp(2.75rem, 5vw + 1rem, 5.25rem),
  major third (1.25) below it
- Weights: body 400, display 700 (300-unit contrast)

## Spacing

4-point named scale (`--space-3xs` to `--space-4xl`). Pages use named tokens,
never raw values. Section padding is deliberately uneven: generous top, tight
bottom before the next rule.

## Motion

Newsprint is 0x: static print metaphor. No reveals, no transforms, no
transitions. Hover, focus, and active states swap colour instantly. Focus rings
appear instantly (never animated). There is no reduced-motion work to do
because nothing moves.

## Microinteractions stance

- Silent success: a scored result is its own confirmation, no toast.
- Focus: instant, 2px outline, 2px offset, `:focus-visible`.
- Button press: instant colour swap, no scale, no lift.
- Validation: Django form errors only, rendered as `errorlist` with a left
  rule pattern plus text (never colour alone).

## CTA voice

- Primary CTA: ink slab - `--color-ink` fill, `--color-paper` text, square
  corners (2px max), verb copy ("Analyze", "Login", "New analysis"). Hover
  swaps fill to `--color-accent` with `--color-accent-ink` text, instantly.
- Secondary CTA: outlined chip - 1px `--color-ink` border, transparent, same
  slab geometry. Hover inverts to ink fill instantly.
- Tertiary: typographic link - 1px underline, colour shift to accent on hover.

## Per-page allowances

- Marketing pages: none exist here.
- App pages MUST NOT use enrichment - function carries the page.
- Content pages: typography only.

## What pages MUST share

- The wordmark (Fraunces, masthead centred).
- The accent colour and its placement (links, focus, primary hover - 5% or
  less per viewport).
- The display + body fonts.
- The CTA voice (slab geometry, padding rhythm).
- Section heading rhythm: `01 / Label` above the heading, single column,
  hairline rule between sections (slop gate 54: never label beside heading).

## What pages MAY differ on

- Single-column head + full-width body, or 7/3 content split with a margin
  note.
- Ledger pages may render data as a table (desktop) or cards (mobile).
- Section count (predict has two numbered sections, auth pages have one).

## Nav and footer

- Nav: N6 newspaper masthead - wordmark, small-caps issue line, link row,
  double rule below. Knobs: issue line below wordmark, wordmark ~2xl, double
  rule.
- Footer: Ft4 dense colophon - one ragged-right monospace paragraph.

## Exports

Drop-in formats for re-using this design system. Source of truth lives in
`django_app/static/css/tokens.css`.

### tokens.css

```css
:root {
  --color-paper:        oklch(97% 0.010 85);
  --color-paper-2:      oklch(94.5% 0.012 85);
  --color-paper-3:      oklch(91% 0.014 84);
  --color-rule:         oklch(84% 0.014 82);
  --color-rule-2:       oklch(58% 0.016 80);
  --color-neutral:      oklch(55% 0.014 80);
  --color-muted:        oklch(44% 0.014 75);
  --color-ink-2:        oklch(30% 0.014 75);
  --color-ink:          oklch(17% 0.012 70);
  --color-accent:       oklch(52% 0.170 27);
  --color-accent-ink:   oklch(97% 0.010 85);
  --color-focus:        oklch(52% 0.170 27);
  --color-error:        oklch(50% 0.190 25);

  --emo-joy:      oklch(64% 0.160 88);
  --emo-anger:    oklch(58% 0.190 27);
  --emo-fear:     oklch(55% 0.170 305);
  --emo-sadness:  oklch(57% 0.120 245);
  --emo-surprise: oklch(65% 0.160 55);
  --emo-disgust:  oklch(60% 0.150 150);
  --emo-neutral:  oklch(64% 0.010 85);

  --font-display: "Fraunces", ui-serif, Georgia, serif;
  --font-body:    "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --font-outlier: "IBM Plex Mono", ui-monospace, monospace;

  --space-3xs: 0.125rem; --space-2xs: 0.25rem; --space-xs: 0.5rem;
  --space-sm: 0.75rem;  --space-md: 1rem;    --space-lg: 1.5rem;
  --space-xl: 2.5rem;   --space-2xl: 4rem;   --space-3xl: 6rem;

  --text-xs: 0.75rem;  --text-sm: 0.875rem; --text-base: 1rem;
  --text-md: 1.25rem;  --text-lg: 1.5625rem; --text-xl: 1.9531rem;
  --text-display: clamp(2.75rem, 5vw + 1rem, 5.25rem);

  --rule-hair: 1px;
  --radius-card: 0; --radius-input: 2px;

  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);
  --dur-micro: 0ms;
}
```

### Tailwind v4 `@theme`

```css
@theme {
  --color-paper:        oklch(97% 0.010 85);
  --color-paper-2:      oklch(94.5% 0.012 85);
  --color-rule:         oklch(84% 0.014 82);
  --color-rule-2:       oklch(58% 0.016 80);
  --color-muted:        oklch(44% 0.014 75);
  --color-ink:          oklch(17% 0.012 70);
  --color-accent:       oklch(52% 0.170 27);

  --font-display: "Fraunces", ui-serif, Georgia, serif;
  --font-body:    "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  --font-outlier: "IBM Plex Mono", ui-monospace, monospace;

  --spacing-3xs: 0.125rem; --spacing-2xs: 0.25rem; --spacing-xs: 0.5rem;
  --spacing-sm: 0.75rem;  --spacing-md: 1rem;    --spacing-lg: 1.5rem;
  --spacing-xl: 2.5rem;   --spacing-2xl: 4rem;   --spacing-3xl: 6rem;

  --radius-card: 0;
}
```

### DTCG `tokens.json`

```json
{
  "color": {
    "paper":  { "$value": "oklch(97% 0.010 85)", "$type": "color" },
    "paper-2": { "$value": "oklch(94.5% 0.012 85)", "$type": "color" },
    "rule":   { "$value": "oklch(84% 0.014 82)", "$type": "color" },
    "rule-2": { "$value": "oklch(58% 0.016 80)", "$type": "color" },
    "muted":  { "$value": "oklch(44% 0.014 75)", "$type": "color" },
    "ink":    { "$value": "oklch(17% 0.012 70)", "$type": "color" },
    "accent": { "$value": "oklch(52% 0.170 27)", "$type": "color" },
    "focus":  { "$value": "oklch(52% 0.170 27)", "$type": "color" }
  },
  "font": {
    "display": { "$value": "Fraunces, ui-serif, Georgia, serif", "$type": "fontFamily" },
    "body":    { "$value": "IBM Plex Sans, ui-sans-serif, system-ui, sans-serif", "$type": "fontFamily" },
    "outlier": { "$value": "IBM Plex Mono, ui-monospace, monospace", "$type": "fontFamily" }
  },
  "space": {
    "sm": { "$value": "0.75rem", "$type": "dimension" },
    "md": { "$value": "1rem", "$type": "dimension" },
    "lg": { "$value": "1.5rem", "$type": "dimension" }
  }
}
```

### shadcn/ui CSS variables

```css
:root {
  --background:          97%   0.010 85;
  --foreground:          17%   0.012 70;
  --card:                94.5% 0.012 85;
  --card-foreground:     17%   0.012 70;
  --primary:             52%   0.170 27;
  --primary-foreground:  97%   0.010 85;
  --muted:               91%   0.014 84;
  --muted-foreground:    44%   0.014 75;
  --border:              84%   0.014 82;
  --input:               58%   0.016 80;
  --ring:                52%   0.170 27;
  --destructive:         50%   0.190 25;
  --radius:              0;
}
```
