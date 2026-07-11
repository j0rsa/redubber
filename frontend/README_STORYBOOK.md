# Storybook for Redubber v2.0

> Interactive component library and design system documentation

## Quick Start

```bash
# Install dependencies (if not already done)
npm install

# Run Storybook
npm run storybook

# Open in browser
# http://localhost:6006
```

## What's Inside

### 📚 Stories (32 total)

- **FileGrid** - 8 variations of video file table
- **PipelineStatus** - 11 pipeline processing states
- **OfflineBanner** - Offline indicator
- **InstallPrompt** - PWA installation prompt
- **ProjectHub** - 5 project management page states
- **ProjectDetail** - 6 project detail page states

### 🎨 Design System

Modern, professional design with:
- Purple-blue gradients (#667eea → #764ba2)
- Smooth animations and hover effects
- Consistent spacing and typography
- Accessible color contrasts (WCAG AA)

### 📖 Documentation

- **[STORYBOOK.md](./STORYBOOK.md)** - Complete guide (2,400+ lines)
  - What Storybook is and why we use it
  - Design system reference
  - Component overview
  - Best practices

- **[DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md)** - Quick reference (850+ lines)
  - Colors, spacing, typography
  - Component patterns
  - Animations
  - Usage examples

- **[COMPONENT_GUIDE.md](./COMPONENT_GUIDE.md)** - API reference (600+ lines)
  - Component hierarchy
  - Props and usage
  - State management
  - Testing guidelines

- **[STORYBOOK_SUMMARY.md](./STORYBOOK_SUMMARY.md)** - Setup summary
  - What was done
  - File changes
  - Verification checklist

## Features

### Modern Components

All components use:
- **CSS Modules** for scoped styling
- **TypeScript** for type safety
- **React Query** for data fetching
- **Zustand** for state management

### Interactive Development

- **Controls Panel** - Adjust props in real-time
- **Docs Tab** - Auto-generated documentation
- **Accessibility Panel** - Check a11y compliance
- **Multiple States** - Test all component variations

### Addons Installed

- `@storybook/addon-docs` - Auto documentation
- `@storybook/addon-a11y` - Accessibility testing
- `@storybook/addon-vitest` - Component testing
- `@chromatic-com/storybook` - Visual regression
- `@storybook/addon-mcp` - MCP integration

## File Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── FileGrid.tsx
│   │   ├── FileGrid.module.css
│   │   ├── FileGrid.stories.tsx
│   │   ├── PipelineStatus.tsx
│   │   ├── PipelineStatus.module.css
│   │   ├── PipelineStatus.stories.tsx
│   │   └── ...
│   ├── pages/
│   │   ├── ProjectHub.tsx
│   │   ├── ProjectHub.stories.tsx
│   │   └── ...
│   ├── styles/
│   │   └── theme.ts
│   └── types/
│       └── index.ts
├── .storybook/
│   ├── main.ts
│   └── preview.tsx
├── STORYBOOK.md
├── DESIGN_SYSTEM.md
├── COMPONENT_GUIDE.md
├── STORYBOOK_SUMMARY.md
└── README_STORYBOOK.md (this file)
```

## Commands

```bash
# Development
npm run storybook          # Run Storybook dev server
npm run build-storybook    # Build static Storybook

# Testing
npm run test              # Run Vitest tests
npx tsc --noEmit         # Type check

# Linting
npm run lint             # Run oxlint
```

## Adding New Stories

### 1. Create Story File

Create `ComponentName.stories.tsx` next to your component:

```typescript
import type { Meta, StoryObj } from '@storybook/react-vite';
import { YourComponent } from './YourComponent';

const meta: Meta<typeof YourComponent> = {
  title: 'Components/YourComponent',
  component: YourComponent,
};

export default meta;
type Story = StoryObj<typeof YourComponent>;

export const Default: Story = {
  args: {
    // Your props here
  },
};
```

### 2. Add Multiple States

```typescript
export const Loading: Story = {
  args: { isLoading: true },
};

export const WithError: Story = {
  args: { error: 'Something went wrong' },
};
```

### 3. Test in Storybook

- Run `npm run storybook`
- Navigate to your component
- Use Controls panel to test props

## Design Tokens

### Colors

```typescript
primary: '#1976d2'         // Blue
success: '#2e7d32'         // Green
error: '#d32f2f'           // Red
gradient: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
```

### Spacing

```typescript
xs: '4px'
sm: '8px'
md: '16px'
lg: '24px'
xl: '32px'
```

### Typography

```typescript
h1: { fontSize: '32px', fontWeight: 700 }
h2: { fontSize: '24px', fontWeight: 600 }
body: { fontSize: '14px', fontWeight: 400 }
```

## Best Practices

### Component Development

1. **Build components in isolation** - Don't run the whole app
2. **Test all states** - Loading, error, empty, success
3. **Use TypeScript** - Type all props
4. **CSS Modules** - Scope styles to prevent conflicts
5. **Accessibility first** - Check with a11y addon

### Story Writing

1. **Name clearly** - `WithProgress`, `EmptyState`, not `Story1`
2. **Show edge cases** - Long text, many items, empty
3. **Document props** - Use descriptions
4. **Keep independent** - Each story should work alone
5. **Use mock data** - Create data generators

## Accessibility

All components should:
- ✅ Have proper color contrast (WCAG AA)
- ✅ Support keyboard navigation
- ✅ Include ARIA labels where needed
- ✅ Use semantic HTML
- ✅ Pass a11y addon checks

## Responsive Design

Components are designed mobile-first:
- 📱 Mobile: < 640px
- 📱 Tablet: < 768px
- 💻 Desktop: < 1024px
- 🖥️ Large: > 1280px

## Resources

- [Storybook Documentation](https://storybook.js.org/docs)
- [Component Story Format](https://storybook.js.org/docs/api/csf)
- [Addons Catalog](https://storybook.js.org/addons)
- [Best Practices](https://storybook.js.org/docs/writing-stories/introduction)

## Troubleshooting

### Stories Not Showing

- Check file naming: `*.stories.tsx`
- Verify `export default meta`
- Restart Storybook

### CSS Not Loading

- Import CSS module: `import styles from './Component.module.css'`
- Check class names match
- Clear cache: `rm -rf node_modules/.vite`

### TypeScript Errors

- Run type check: `npx tsc --noEmit`
- Check imports from `../types`
- Verify interface matches

## Contributing

When adding new components:

1. Create component file (`.tsx`)
2. Create CSS module (`.module.css`)
3. Create stories (`.stories.tsx`)
4. Update documentation
5. Test accessibility
6. Submit PR

## Support

- 📖 Check [STORYBOOK.md](./STORYBOOK.md) for detailed guide
- 🎨 See [DESIGN_SYSTEM.md](./DESIGN_SYSTEM.md) for styling
- 📚 Read [COMPONENT_GUIDE.md](./COMPONENT_GUIDE.md) for usage

---

**Status**: ✅ Complete and verified  
**Version**: Storybook 10.4.6  
**Last Updated**: 2026-07-07

🎉 Happy component building! 🎨
