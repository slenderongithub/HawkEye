/**
 * HawkEye design tokens — the single source of truth for the app's look.
 *
 * The palette extends the existing operator dashboard (server/static/index.html)
 * so the mobile app reads as the same product: a dark "command center" surface,
 * one blue brand accent, and a disciplined set of semantic colors that map 1:1
 * to the backend's Severity / ActionType / IncidentStatus enums.
 *
 * Design intent: calm under load. Restrained accent usage, real typographic
 * hierarchy, soft depth (hairline borders + faint elevation), and motion that
 * is quick and spring-based rather than showy.
 */

export const palette = {
  // Base surfaces (darkest → most elevated)
  bg: '#0B0E14',
  surface: '#141926',
  surfaceElevated: '#1B2230',
  surfaceHover: '#212a3a',
  hairline: '#242C3A',
  hairlineStrong: '#313B4E',

  // Text
  text: '#E6E9EF',
  textSecondary: '#8B93A7',
  textTertiary: '#5C6478',
  textInverse: '#04121F',

  // Brand accent (actions, LIVE, active states)
  accent: '#3EA6FF',
  accentDim: 'rgba(62,166,255,0.14)',
  accentBorder: 'rgba(62,166,255,0.35)',

  // Semantic (aligned to backend enums)
  green: '#37D67A', // ok / present / severity.low
  greenDim: 'rgba(55,214,122,0.15)',
  amber: '#FFB020', // warn / severity.medium
  amberDim: 'rgba(255,176,32,0.15)',
  red: '#FF4D5E', // alert / severity.high + critical
  redDim: 'rgba(255,77,94,0.15)',
  redSolid: '#FF4D5E',
  purple: '#A074FF', // AI reasoning / info
  purpleDim: 'rgba(160,116,255,0.15)',
  cyan: '#37D3D3',
  cyanDim: 'rgba(55,211,211,0.15)',

  // Pure
  black: '#000000',
  white: '#FFFFFF',
  overlay: 'rgba(6,9,15,0.72)',
} as const;

/** Vertical accent gradient for hero moments (kept subtle). */
export const gradients = {
  brand: ['#3EA6FF', '#5B8CFF'] as const,
  alert: ['#FF4D5E', '#FF7A45'] as const,
  surfaceGlow: ['rgba(62,166,255,0.10)', 'rgba(62,166,255,0)'] as const,
  scrim: ['rgba(11,14,20,0)', 'rgba(11,14,20,0.85)'] as const,
};

/** 4px baseline spacing scale. */
export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
  xxxl: 40,
} as const;

export const radii = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 22,
  pill: 999,
} as const;

export const typography = {
  // Sizes
  hero: { fontSize: 52, lineHeight: 56, fontWeight: '800' as const, letterSpacing: -1 },
  display: { fontSize: 28, lineHeight: 34, fontWeight: '800' as const, letterSpacing: 0.2 },
  title: { fontSize: 22, lineHeight: 28, fontWeight: '700' as const },
  heading: { fontSize: 17, lineHeight: 22, fontWeight: '700' as const },
  body: { fontSize: 15, lineHeight: 21, fontWeight: '500' as const },
  bodyStrong: { fontSize: 15, lineHeight: 21, fontWeight: '700' as const },
  caption: { fontSize: 13, lineHeight: 18, fontWeight: '500' as const },
  micro: { fontSize: 11, lineHeight: 14, fontWeight: '700' as const, letterSpacing: 0.4 },
  label: {
    fontSize: 11,
    lineHeight: 14,
    fontWeight: '700' as const,
    letterSpacing: 0.8,
    textTransform: 'uppercase' as const,
  },
} as const;

/** Soft, layered shadow presets (iOS + Android elevation). */
export const shadows = {
  card: {
    shadowColor: '#000',
    shadowOpacity: 0.35,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 8 },
    elevation: 6,
  },
  floating: {
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 12 },
    elevation: 12,
  },
} as const;

/**
 * Motion tokens. Springs are the default; durations are for fades/opacity only.
 * Tuned to feel quick and physical — never bouncy or slow.
 */
export const motion = {
  spring: { damping: 18, stiffness: 220, mass: 0.9 },
  springSoft: { damping: 20, stiffness: 140, mass: 1 },
  pressScale: 0.97,
  duration: { fast: 140, base: 220, slow: 360 },
  stagger: 45, // ms between list-item entrances
} as const;

export type Palette = typeof palette;
