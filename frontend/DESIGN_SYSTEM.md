# Redubber Design System

Quick reference guide for consistent UI design across the Redubber v2.0 frontend.

## Colors

### Primary Palette

```css
--primary: #1976d2;           /* Blue - Primary actions */
--primary-hover: #1565c0;     /* Darker blue - Hover states */
--secondary: #dc004e;         /* Pink - Secondary actions */
--success: #2e7d32;           /* Green - Success states */
--warning: #ed6c02;           /* Orange - Warning states */
--error: #d32f2f;             /* Red - Error states */
```

### Neutral Palette

```css
--background: #ffffff;        /* White - Main background */
--background-dark: #f5f5f5;   /* Light gray - Secondary background */
--text: #212121;              /* Dark gray - Primary text */
--text-secondary: #757575;    /* Medium gray - Secondary text */
--border: #e0e0e0;            /* Light gray - Borders */
--shadow: rgba(0, 0, 0, 0.1); /* Shadow color */
```

### Gradients

```css
/* Primary Gradient - Buttons, Headers */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Progress Bar Gradient */
background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
```

## Typography

### Font Family

```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
```

### Type Scale

| Name | Size | Weight | Usage |
|------|------|--------|-------|
| H1 | 32px | 700 | Page titles |
| H2 | 24px | 600 | Section headers |
| H3 | 20px | 600 | Card headers |
| Body | 14px | 400 | Body text |
| Small | 12px | 400 | Captions, metadata |

## Spacing

Use consistent spacing values:

```css
--spacing-xs: 4px;    /* Tight spacing */
--spacing-sm: 8px;    /* Small gaps */
--spacing-md: 16px;   /* Default spacing */
--spacing-lg: 24px;   /* Section spacing */
--spacing-xl: 32px;   /* Large gaps */
```

## Border Radius

```css
--radius-sm: 4px;     /* Buttons, inputs */
--radius-md: 8px;     /* Cards, containers */
--radius-lg: 12px;    /* Large cards */
```

## Shadows

```css
/* Card Shadow */
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);

/* Hover Shadow */
box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
```

## Components

### Buttons

#### Primary Button

```css
.button {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.2s;
}

.button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

#### Secondary Button

```css
.button-secondary {
  padding: 8px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  background: white;
  color: #212121;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.button-secondary:hover {
  border-color: #1976d2;
  background: #f8f9ff;
}
```

### Cards

```css
.card {
  background: white;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
```

### Badges

```css
.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.badge.pending {
  background: #f5f5f5;
  color: #757575;
}

.badge.running {
  background: #e3f2fd;
  color: #1976d2;
}

.badge.completed {
  background: #e8f5e9;
  color: #2e7d32;
}

.badge.error {
  background: #ffebee;
  color: #d32f2f;
}
```

### Progress Bar

```css
.progress-bar {
  width: 100%;
  height: 12px;
  background: #f0f0f0;
  border-radius: 6px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
  border-radius: 6px;
}
```

### Tables

```css
.table {
  width: 100%;
  border-collapse: collapse;
}

.table thead {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-weight: 600;
}

.table th {
  padding: 16px;
  text-align: left;
}

.table tr {
  border-bottom: 1px solid #f0f0f0;
  transition: background-color 0.2s;
}

.table tr:hover {
  background-color: #f8f9ff;
}

.table td {
  padding: 16px;
}
```

## Animations

### Pulse Animation (for loading states)

```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}

.loading {
  animation: pulse 2s infinite;
}
```

### Fade In

```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.fade-in {
  animation: fadeIn 0.3s ease-in;
}
```

### Slide Up

```css
@keyframes slideUp {
  from {
    transform: translateY(20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.slide-up {
  animation: slideUp 0.3s ease-out;
}
```

## States

### Hover States

- Transform: `translateY(-2px)`
- Shadow: `0 4px 12px rgba(102, 126, 234, 0.4)`
- Border color change
- Background lightening

### Disabled States

- Opacity: `0.5`
- Cursor: `not-allowed`
- No hover effects

### Focus States

```css
.input:focus {
  outline: none;
  border-color: #1976d2;
  box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
}
```

### Loading States

- Pulse animation
- Reduced opacity
- Spinner icon (if applicable)

## Icons

Use emoji icons for quick implementation:

- 🔊 Audio
- 📝 Transcript
- 🎙️ TTS
- ✅ Complete
- ⚠️ Warning
- ❌ Error
- 🌐 Language
- 📁 File

## Accessibility

### Color Contrast

- Primary text on white: `#212121` (AA compliant)
- Secondary text on white: `#757575` (AA compliant)
- White text on primary: Always use white text on colored backgrounds

### Focus Indicators

Always provide visible focus indicators:

```css
.focusable:focus {
  outline: 2px solid #1976d2;
  outline-offset: 2px;
}
```

### Screen Reader Text

```css
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}
```

## Responsive Breakpoints

```css
/* Mobile */
@media (max-width: 640px) { ... }

/* Tablet */
@media (max-width: 768px) { ... }

/* Desktop */
@media (max-width: 1024px) { ... }

/* Large Desktop */
@media (min-width: 1280px) { ... }
```

## Usage Examples

### Card with Gradient Header

```tsx
<div className={styles.card}>
  <div className={styles.gradientHeader}>
    <h2>Header Text</h2>
  </div>
  <div className={styles.cardBody}>
    Content here
  </div>
</div>
```

### Status Badge

```tsx
<span className={`${styles.badge} ${styles.running}`}>
  Running
</span>
```

### Action Button with Hover

```tsx
<button className={styles.actionButton}>
  Start Processing
</button>
```

## Best Practices

1. **Use CSS Modules** - Scope styles to prevent conflicts
2. **Follow naming conventions** - Use kebab-case for CSS classes
3. **Maintain consistency** - Use design tokens from theme.ts
4. **Test accessibility** - Check with Storybook's a11y addon
5. **Mobile-first** - Design for small screens first
6. **Smooth transitions** - Use 0.2s-0.3s for most transitions
7. **Semantic colors** - Use success/warning/error appropriately
8. **White space** - Don't be afraid of padding/margin

---

**Version**: 1.0  
**Last Updated**: 2026-07-07  
**Maintained by**: Redubber Team
