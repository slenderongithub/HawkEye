/**
 * HawkEye design tokens — the single source of truth for the app's look.
 *
 * The system is Apple's design language expressed on its **dark-tile chassis**.
 * HawkEye is a live monitoring surface dominated by a black video feed, so the
 * near-black product tiles (#272729 / #2a2a2c / #252527) are the right register,
 * not the white/parchment marketing one. Everything else follows the language as
 * written:
 *
 *   - **One accent, and only one.** Sky Link Blue (#2997ff) carries every
 *     interactive element. Action Blue (#0066cc) is the light-surface variant and
 *     would disappear against these tiles, so it is deliberately absent.
 *   - **Elevation is surface colour, not shadow.** Apple uses exactly one
 *     drop-shadow in the whole system and reserves it for product photography
 *     resting on a surface. Here that is the live feed — the app's "product" —
 *     and nothing else. Cards separate by a 1px hairline and a micro-step change
 *     in surface lightness.
 *   - **No decorative gradients.** Atmosphere comes from content. The only
 *     gradient left is a legibility scrim for text over video, which is
 *     functional rather than decorative.
 *   - **Body copy is 17px, and the weight ladder is 300 / 400 / 600 / 700.**
 *     Weight 500 is deliberately absent from the language; display sizes carry
 *     negative tracking for the signature tight headline cadence.
 *
 * Status colours (severity, dispatch state) are the one place the language is
 * silent — it has no opinion on incident semantics — so those use Apple's own
 * dark-mode system colours, which keeps them native to the palette.
 */

export const palette = {
  // Base surfaces, darkest → most elevated. Apple's dark tiles are siblings
  // rather than a ladder; ordering them by lightness gives depth without shadow.
  bg: '#1d1d1f',
  surface: '#272729',
  surfaceElevated: '#2a2a2c',
  surfaceHover: '#313134',
  hairline: 'rgba(255,255,255,0.08)',
  hairlineStrong: 'rgba(255,255,255,0.16)',

  // Text on dark tiles.
  text: '#ffffff',
  textSecondary: '#cccccc',
  textTertiary: '#7a7a7a',
  textInverse: '#1d1d1f',

  // The single interactive colour. Sky Link Blue — the dark-surface variant.
  accent: '#2997ff',
  accentDim: 'rgba(41,151,255,0.14)',
  accentBorder: 'rgba(41,151,255,0.35)',

  // Semantic status (aligned to backend enums). Apple dark-mode system colours.
  // Each has a Dim fill and a Border so status chrome never needs a literal
  // rgba() at the call site — that is how a palette silently forks.
  green: '#30d158', // ok / present / severity.low
  greenDim: 'rgba(48,209,88,0.15)',
  greenBorder: 'rgba(48,209,88,0.35)',
  amber: '#ff9f0a', // warn / severity.medium
  amberDim: 'rgba(255,159,10,0.15)',
  amberBorder: 'rgba(255,159,10,0.35)',
  red: '#ff453a', // alert / severity.high + critical
  redDim: 'rgba(255,69,58,0.15)',
  redBorder: 'rgba(255,69,58,0.5)',
  redSolid: '#ff453a',
  purple: '#bf5af2', // AI reasoning / info
  purpleDim: 'rgba(191,90,242,0.15)',
  cyan: '#40c8e0',
  cyanDim: 'rgba(64,200,224,0.15)',

  // Pure
  black: '#000000',
  white: '#ffffff',
  overlay: 'rgba(0,0,0,0.72)',
  /** Frosted bar fill — the language's sub-nav/sticky-bar treatment, on dark. */
  surfaceTranslucent: 'rgba(39,39,41,0.75)',
  /** Neutral chip over photography (the guide's translucent control chip). */
  chipTranslucent: 'rgba(210,210,215,0.16)',
} as const;

/**
 * The only gradient in the system: a scrim so text stays legible over video.
 * Decorative gradients (brand washes, alert ramps, glows) are not part of the
 * language — depth comes from surface changes and photography.
 */
export const gradients = {
  scrim: ['rgba(29,29,31,0)', 'rgba(29,29,31,0.85)'] as const,
};

/** Apple's spacing scale. Base unit 8, with 17 as the body-rhythm step. */
export const spacing = {
  xxs: 4,
  xs: 8,
  sm: 12,
  md: 17,
  lg: 24,
  xl: 32,
  xxl: 48,
  section: 80,
} as const;

/**
 * Radius grammar. Each value means something: `sm` for compact utility chrome,
 * `md` for pearl capsules, `lg` for cards, `pill` for anything that reads as an
 * action. Nothing in between — mixing radii is what makes a UI look improvised.
 */
export const radii = {
  xs: 5,
  sm: 8,
  md: 11,
  lg: 18,
  pill: 9999,
} as const;

/**
 * Type ladder. Sizes, weights and tracking come straight from the language;
 * line heights are the guide's multipliers resolved to px for React Native.
 * SF Pro is the system font on Apple platforms, so `System` resolves to the real
 * thing on iOS/macOS and to a sane sans elsewhere.
 */
export const typography = {
  hero: { fontSize: 56, lineHeight: 60, fontWeight: '600' as const, letterSpacing: -0.28 },
  display: { fontSize: 40, lineHeight: 44, fontWeight: '600' as const, letterSpacing: 0 },
  title: { fontSize: 28, lineHeight: 32, fontWeight: '600' as const, letterSpacing: 0.196 },
  heading: { fontSize: 21, lineHeight: 25, fontWeight: '600' as const, letterSpacing: 0.231 },
  body: { fontSize: 17, lineHeight: 25, fontWeight: '400' as const, letterSpacing: -0.374 },
  bodyStrong: { fontSize: 17, lineHeight: 21, fontWeight: '600' as const, letterSpacing: -0.374 },
  lead: { fontSize: 24, lineHeight: 36, fontWeight: '300' as const, letterSpacing: 0 },
  caption: { fontSize: 14, lineHeight: 20, fontWeight: '400' as const, letterSpacing: -0.224 },
  captionStrong: { fontSize: 14, lineHeight: 18, fontWeight: '600' as const, letterSpacing: -0.224 },
  micro: { fontSize: 12, lineHeight: 14, fontWeight: '400' as const, letterSpacing: -0.12 },
  label: {
    fontSize: 12,
    lineHeight: 14,
    fontWeight: '600' as const,
    letterSpacing: -0.12,
    textTransform: 'uppercase' as const,
  },
} as const;

/**
 * The single drop-shadow in the system, reserved for the live feed — this app's
 * equivalent of a product render resting on a surface. Do not add shadows to
 * cards, buttons, chips or text: alternate the surface instead.
 */
export const shadows = {
  product: {
    shadowColor: '#000',
    shadowOpacity: 0.22,
    shadowRadius: 30,
    shadowOffset: { width: 3, height: 5 },
    elevation: 8,
  },
} as const;

/**
 * Motion tokens. `pressScale` is the language's system-wide press interaction —
 * every button scales to 0.95 when pressed, nothing else.
 */
export const motion = {
  spring: { damping: 18, stiffness: 220, mass: 0.9 },
  springSoft: { damping: 20, stiffness: 140, mass: 1 },
  pressScale: 0.95,
  duration: { fast: 140, base: 220, slow: 360 },
  stagger: 45, // ms between list-item entrances
} as const;

export type Palette = typeof palette;
