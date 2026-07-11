# Redubber v2.0 Design System

Modern, accessible design system with automatic light/dark mode support based on system preferences.

## Philosophy

- **Desktop-first**: Designed for professional workflows, scales down to mobile
- **System-aware**: Respects user's OS dark mode preference via `prefers-color-scheme`
- **CSS Variables**: All tokens defined as CSS custom properties for easy theming
- **Type-safe**: TypeScript theme object provides compile-time safety
- **Accessible**: WCAG 2.1 AA compliant color contrasts

## File Structure

```
frontend/src/styles/
├── variables.css      # CSS custom properties (design tokens)
└── theme.ts          # TypeScript theme object

frontend/src/index.css # Global styles + variable imports
```

## Usage

### In CSS Modules

Use CSS custom properties directly:

```css
.myComponent {
  background: var(--color-bg-elevated);
  color: var(--color-text-primary);
  padding: var(--spacing-md);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-md);
}

.button {
  background: var(--color-primary);
  color: white;
  transition: var(--transition-base);
}

.button:hover {
  background: var(--color-primary-hover);
}
```

### In TypeScript/JSX (Inline Styles)

Use the theme object:

```typescript
import { theme } from '@/styles/theme';

const MyComponent = () => (
  <div style={{
    background: theme.colors.bgElevated,
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
  }}>
    Content
  </div>
);
```

### Detect Color Scheme

```typescript
import { getColorScheme, watchColorScheme } from '@/styles/theme';

// Get current scheme
const scheme = getColorScheme(); // 'light' | 'dark'

// Listen to changes
useEffect(() => {
  const unwatch = watchColorScheme((scheme) => {
    console.log('Color scheme changed to:', scheme);
  });
  return unwatch;
}, []);
```

## Color Palette

### Primary Colors

| Variable | Light Mode | Dark Mode | Usage |
|----------|-----------|-----------|-------|
| `--color-primary` | `#1976d2` | `#42a5f5` | Primary actions, links, focus states |
| `--color-primary-hover` | `#1565c0` | `#64b5f6` | Hover states |
| `--color-primary-light` | `#e3f2fd` | `rgba(66,165,245,0.15)` | Backgrounds, selected states |
| `--color-primary-dark` | `#0d47a1` | `#1976d2` | Dark accents |

### Status Colors

| Variable | Light Mode | Dark Mode | Usage |
|----------|-----------|-----------|-------|
| `--color-success` | `#2e7d32` | `#66bb6a` | Success messages, completed states |
| `--color-warning` | `#ed6c02` | `#ffa726` | Warnings, alerts |
| `--color-error` | `#d32f2f` | `#ef5350` | Errors, destructive actions |
| `--color-info` | `#0288d1` | `#29b6f6` | Informational messages |

### Background Colors

| Variable | Light Mode | Dark Mode | Usage |
|----------|-----------|-----------|-------|
| `--color-bg-primary` | `#ffffff` | `#121212` | Main background |
| `--color-bg-secondary` | `#f5f5f5` | `#1e1e1e` | Secondary backgrounds, panels |
| `--color-bg-tertiary` | `#fafafa` | `#2a2a2a` | Tertiary backgrounds, code blocks |
| `--color-bg-elevated` | `#ffffff` | `#1e1e1e` | Cards, modals, elevated surfaces |

### Text Colors

| Variable | Light Mode | Dark Mode | Usage |
|----------|-----------|-----------|-------|
| `--color-text-primary` | `#212121` | `#ffffff` | Primary text, headings |
| `--color-text-secondary` | `#757575` | `#b0b0b0` | Secondary text, descriptions |
| `--color-text-tertiary` | `#9e9e9e` | `#757575` | Tertiary text, hints |
| `--color-text-disabled` | `#bdbdbd` | `#4a4a4a` | Disabled states |

### Borders & Dividers

| Variable | Light Mode | Dark Mode | Usage |
|----------|-----------|-----------|-------|
| `--color-border` | `#e0e0e0` | `#3a3a3a` | Standard borders |
| `--color-border-light` | `#f5f5f5` | `#2a2a2a` | Subtle dividers |
| `--color-divider` | `#e0e0e0` | `#3a3a3a` | Section dividers |

## Spacing Scale

Based on 4px base unit:

| Variable | Value | Usage |
|----------|-------|-------|
| `--spacing-xs` | `4px` | Tight spacing |
| `--spacing-sm` | `8px` | Small gaps |
| `--spacing-md` | `16px` | Standard spacing |
| `--spacing-lg` | `24px` | Large spacing |
| `--spacing-xl` | `32px` | Extra large spacing |
| `--spacing-2xl` | `48px` | Section spacing |

## Border Radius

| Variable | Value | Usage |
|----------|-------|-------|
| `--radius-sm` | `4px` | Small elements |
| `--radius-md` | `8px` | Buttons, inputs |
| `--radius-lg` | `12px` | Cards, panels |
| `--radius-xl` | `16px` | Large surfaces |
| `--radius-full` | `9999px` | Pills, avatars |

## Typography

### Font Families

```css
--font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif
--font-family-mono: 'Courier New', Monaco, Consolas, monospace
```

### Font Sizes

| Variable | Value | Usage |
|----------|-------|-------|
| `--font-size-xs` | `11px` | Tiny labels |
| `--font-size-sm` | `12px` | Small text, captions |
| `--font-size-base` | `14px` | Body text |
| `--font-size-md` | `15px` | Medium text |
| `--font-size-lg` | `16px` | Large text |
| `--font-size-xl` | `20px` | Headings |
| `--font-size-2xl` | `24px` | Large headings |
| `--font-size-3xl` | `32px` | Page titles |

### Font Weights

| Variable | Value | Usage |
|----------|-------|-------|
| `--font-weight-normal` | `400` | Body text |
| `--font-weight-medium` | `500` | Emphasis |
| `--font-weight-semibold` | `600` | Headings |
| `--font-weight-bold` | `700` | Strong emphasis |

### Line Heights

| Variable | Value | Usage |
|----------|-------|-------|
| `--line-height-tight` | `1.2` | Headings |
| `--line-height-normal` | `1.5` | Body text |
| `--line-height-relaxed` | `1.6` | Long-form content |

## Shadows

| Variable | Value | Usage |
|----------|-------|-------|
| `--shadow-sm` | Light/dark aware | Subtle elevation |
| `--shadow-md` | Light/dark aware | Standard elevation |
| `--shadow-lg` | Light/dark aware | High elevation |
| `--shadow-primary` | Light/dark aware | Primary button shadow |

## Transitions

| Variable | Value | Usage |
|----------|-------|-------|
| `--transition-fast` | `150ms ease-in-out` | Quick interactions |
| `--transition-base` | `200ms ease-in-out` | Standard animations |
| `--transition-slow` | `300ms ease-in-out` | Slow animations |

## Z-Index Scale

| Variable | Value | Usage |
|----------|-------|-------|
| `--z-dropdown` | `1000` | Dropdown menus |
| `--z-sticky` | `1020` | Sticky headers |
| `--z-fixed` | `1030` | Fixed elements |
| `--z-modal-backdrop` | `1040` | Modal backdrops |
| `--z-modal` | `1050` | Modal dialogs |
| `--z-popover` | `1060` | Popovers |
| `--z-tooltip` | `1070` | Tooltips |

## Responsive Design

### Breakpoints

Desktop-first approach:

```css
/* Desktop (default): 1024px+ */
/* Tablet: max-width: 1024px */
@media (max-width: 1024px) { }

/* Mobile: max-width: 768px */
@media (max-width: 768px) { }

/* Small mobile: max-width: 480px */
@media (max-width: 480px) { }
```

### Design Principles

1. **Desktop-first layout**: Components default to spacious desktop layouts
2. **Progressive enhancement**: Mobile gets optimized touch targets (48px+)
3. **Fluid typography**: Font sizes scale appropriately per breakpoint
4. **Touch-friendly**: Mobile buttons/inputs sized for finger taps
5. **Readable**: Sufficient contrast at all breakpoints

## Accessibility

### Color Contrast

All color combinations meet WCAG 2.1 AA standards:
- Normal text: 4.5:1 minimum
- Large text (18px+): 3:1 minimum
- UI components: 3:1 minimum

### Focus States

All interactive elements use visible focus indicators:

```css
.button:focus-visible {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}
```

### Reduced Motion

Respect user's motion preferences:

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Best Practices

### ✅ Do

- Use CSS custom properties in CSS modules
- Use theme object for inline styles
- Respect system color scheme
- Test both light and dark modes
- Use semantic color names (primary, not blue)
- Follow spacing scale

### ❌ Don't

- Hardcode hex colors
- Use pixel values instead of variables
- Override system preferences without user control
- Use colors outside the palette
- Create custom spacing values

## Storybook Integration

View components in both light and dark modes:

1. Open Storybook: `npm run storybook`
2. Use the background selector in toolbar
3. Test desktop, tablet, and mobile viewports
4. Check accessibility tab for contrast issues

## Migration Guide

Converting old hardcoded styles:

```css
/* Before */
.old {
  background: #ffffff;
  color: #212121;
  padding: 16px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}

/* After */
.new {
  background: var(--color-bg-elevated);
  color: var(--color-text-primary);
  padding: var(--spacing-md);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
```

## Resources

- [CSS Custom Properties (MDN)](https://developer.mozilla.org/en-US/docs/Web/CSS/--*)
- [prefers-color-scheme (MDN)](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [Material Design System](https://m3.material.io/)
