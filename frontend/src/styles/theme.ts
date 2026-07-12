/**
 * Modern design system for Redubber v2.0
 *
 * This theme object provides TypeScript-friendly access to CSS custom properties
 * defined in variables.css. The actual values adapt to light/dark mode via
 * prefers-color-scheme media queries.
 *
 * Usage:
 * - In CSS modules: Use var(--color-primary) directly
 * - In inline styles: Use theme.colors.primary
 * - In TypeScript: Import theme for type-safe access
 */

export const theme = {
  colors: {
    // Primary colors
    primary: 'var(--color-primary)',
    primaryHover: 'var(--color-primary-hover)',
    primaryLight: 'var(--color-primary-light)',
    primaryDark: 'var(--color-primary-dark)',

    // Secondary & Status colors
    secondary: 'var(--color-secondary)',
    secondaryHover: 'var(--color-secondary-hover)',
    success: 'var(--color-success)',
    successLight: 'var(--color-success-light)',
    warning: 'var(--color-warning)',
    warningLight: 'var(--color-warning-light)',
    error: 'var(--color-error)',
    errorLight: 'var(--color-error-light)',
    info: 'var(--color-info)',
    infoLight: 'var(--color-info-light)',

    // Background colors
    bgPrimary: 'var(--color-bg-primary)',
    bgSecondary: 'var(--color-bg-secondary)',
    bgTertiary: 'var(--color-bg-tertiary)',
    bgElevated: 'var(--color-bg-elevated)',

    // Text colors
    textPrimary: 'var(--color-text-primary)',
    textSecondary: 'var(--color-text-secondary)',
    textTertiary: 'var(--color-text-tertiary)',
    textDisabled: 'var(--color-text-disabled)',

    // Border & Divider
    border: 'var(--color-border)',
    borderLight: 'var(--color-border-light)',
    divider: 'var(--color-divider)',
  },

  spacing: {
    xs: 'var(--spacing-xs)',
    sm: 'var(--spacing-sm)',
    md: 'var(--spacing-md)',
    lg: 'var(--spacing-lg)',
    xl: 'var(--spacing-xl)',
    '2xl': 'var(--spacing-2xl)',
  },

  borderRadius: {
    sm: 'var(--radius-sm)',
    md: 'var(--radius-md)',
    lg: 'var(--radius-lg)',
    xl: 'var(--radius-xl)',
    full: 'var(--radius-full)',
  },

  typography: {
    fontFamily: 'var(--font-family)',
    fontFamilyMono: 'var(--font-family-mono)',

    fontSize: {
      xs: 'var(--font-size-xs)',
      sm: 'var(--font-size-sm)',
      base: 'var(--font-size-base)',
      md: 'var(--font-size-md)',
      lg: 'var(--font-size-lg)',
      xl: 'var(--font-size-xl)',
      '2xl': 'var(--font-size-2xl)',
      '3xl': 'var(--font-size-3xl)',
    },

    fontWeight: {
      normal: 'var(--font-weight-normal)',
      medium: 'var(--font-weight-medium)',
      semibold: 'var(--font-weight-semibold)',
      bold: 'var(--font-weight-bold)',
    },

    lineHeight: {
      tight: 'var(--line-height-tight)',
      normal: 'var(--line-height-normal)',
      relaxed: 'var(--line-height-relaxed)',
    },
  },

  shadow: {
    sm: 'var(--shadow-sm)',
    md: 'var(--shadow-md)',
    lg: 'var(--shadow-lg)',
    primary: 'var(--shadow-primary)',
  },

  transition: {
    fast: 'var(--transition-fast)',
    base: 'var(--transition-base)',
    slow: 'var(--transition-slow)',
  },

  zIndex: {
    dropdown: 'var(--z-dropdown)',
    sticky: 'var(--z-sticky)',
    fixed: 'var(--z-fixed)',
    modalBackdrop: 'var(--z-modal-backdrop)',
    modal: 'var(--z-modal)',
    popover: 'var(--z-popover)',
    tooltip: 'var(--z-tooltip)',
  },
} as const;

export type Theme = typeof theme;

/**
 * Helper to get current color scheme preference
 */
export const getColorScheme = (): 'light' | 'dark' => {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
};

/**
 * Listen to color scheme changes
 */
export const watchColorScheme = (callback: (scheme: 'light' | 'dark') => void) => {
  if (typeof window === 'undefined') return () => {};

  const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  const handler = (e: MediaQueryListEvent) => callback(e.matches ? 'dark' : 'light');

  mediaQuery.addEventListener('change', handler);
  return () => mediaQuery.removeEventListener('change', handler);
};
