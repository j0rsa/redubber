# Voice Refinement Storybook Stories - Developer Guide

## Quick Start

```bash
cd frontend
npm run storybook
```

Then navigate to the "Components" section in the Storybook sidebar.

## Story Organization

### Main Workflow Component
- **VoiceRefinement** - Full voice refinement workflow container
  - Path: `Components/VoiceRefinement`
  - 19 stories covering all workflow states

### Sub-Components
- **SegmentSelector** - Audio segment selection
  - Path: `Components/VoiceRefinement/SegmentSelector`
  - 17 stories covering selection states

- **VoiceAnalyzer** - AI voice analysis and instruction generation
  - Path: `Components/VoiceRefinement/VoiceAnalyzer`
  - 18 stories covering analysis workflow

- **VoicePreviewGrid** - Voice preview comparison grid
  - Path: `Components/VoiceRefinement/VoicePreviewGrid`
  - 19 stories covering preview states

### Audio Players
- **AudioPlayer** - Basic audio playback
  - Path: `Components/AudioPlayer`
  - 15 stories (8 original + 7 enhanced)

- **AudioPlayerWithWaveform** - Audio with waveform visualization
  - Path: `Components/AudioPlayerWithWaveform`
  - 14 stories (7 original + 7 enhanced)

## Common Story Patterns

### Testing Component States

#### Default State
```typescript
export const Default: Story = {
  args: {
    // Minimal props for initial state
  },
};
```

#### Loading State
```typescript
export const Loading: Story = {
  args: {
    isLoading: true,
    // Other relevant props
  },
};
```

#### Error State
```typescript
export const ErrorState: Story = {
  args: {
    error: 'Error message here',
    // Other relevant props
  },
};
```

### Testing Responsive Design

#### Mobile
```typescript
export const Mobile: Story = {
  args: { /* ... */ },
  parameters: {
    viewport: {
      defaultViewport: 'mobile1',
    },
  },
};
```

#### Tablet
```typescript
export const Tablet: Story = {
  args: { /* ... */ },
  parameters: {
    viewport: {
      defaultViewport: 'tablet',
    },
  },
};
```

### Testing Themes

#### Dark Background
```typescript
export const DarkBackground: Story = {
  args: { /* ... */ },
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};
```

## Mock Data Reference

### Creating Mock Segments
```typescript
const createSegment = (
  id: string,
  videoFilename: string,
  startTime: number,
  endTime: number,
  originalText: string,
  translatedText: string
): TranscriptionSegment => ({
  id,
  video_filename: videoFilename,
  start_time: startTime,
  end_time: endTime,
  duration: endTime - startTime,
  original_text: originalText,
  translated_text: translatedText,
  audio_url: `https://example.com/audio/${id}.mp3`,
});
```

### Mock Voice Instructions
```typescript
const mockInstructions: VoiceInstructions = {
  text: 'Professional, clear, and engaging tone...',
  detected_characteristics: {
    tone: 'Professional and educational',
    pace: 'Moderate, clear articulation',
    emotion: 'Engaged and enthusiastic',
    style: 'Tutorial presenter',
  },
  generation_id: 1,
};
```

### Mock Voice Previews
```typescript
const mockPreviews: VoicePreview[] = [
  {
    voice: 'alloy',
    audio_url: 'https://example.com/preview/alloy.mp3',
    duration_ms: 8500,
    cached: true,
  },
  // ... more previews
];
```

## Available Voice Options

| Voice ID | Name | Description | Gender |
|----------|------|-------------|--------|
| alloy | Alloy | Neutral, balanced voice | neutral |
| echo | Echo | Male, clear and articulate | male |
| fable | Fable | British, expressive | male |
| onyx | Onyx | Deep, authoritative male | male |
| nova | Nova | Warm, engaging female | female |
| shimmer | Shimmer | Soft, gentle female | female |

## Story Naming Conventions

- `Default` - Initial/default state
- `Loading` / `Analyzing` / `Generating` - Loading states
- `Error*` - Error states (e.g., `ErrorAnalysisFailed`)
- `Empty*` / `No*` - Empty states (e.g., `EmptyProject`, `NoSegments`)
- `*Selected` - Selected/active states
- `Playing*` - Audio playing states
- `Mobile` / `Tablet` / `Desktop*` - Viewport variations
- `Dark*` - Dark theme variations
- `Compact` - Compact/minimal variations
- `Long*` / `Short*` - Content length variations
- `Many*` / `Few*` - Quantity variations

## Interactive Stories

Some stories include interactive elements using Storybook's `play` function:

```typescript
export const EditingInstructions: Story = {
  args: { /* ... */ },
  play: async ({ canvasElement }) => {
    // Simulate clicking edit button
    const editButton = canvasElement.querySelector('button');
    if (editButton) {
      editButton.click();
    }
  },
};
```

## Testing Checklist

When creating new stories, ensure you cover:

- [ ] Default state
- [ ] Loading states
- [ ] Error states
- [ ] Empty states
- [ ] Success/completed states
- [ ] Mobile viewport
- [ ] Tablet viewport (if applicable)
- [ ] Desktop viewport
- [ ] Dark background
- [ ] Edge cases (long text, many items, etc.)
- [ ] Interactive states (selected, playing, editing)

## Callback Testing

All interactive callbacks log to the console for testing:

```typescript
args: {
  onSegmentSelect: (segment) => console.log('Segment selected:', segment),
  onAnalyze: async () => {
    console.log('Analyze clicked');
    await new Promise(resolve => setTimeout(resolve, 2000));
  },
}
```

Open the browser console to see callback logs when interacting with stories.

## Audio URLs

Stories use placeholder audio URLs from example.com:
- These URLs won't load actual audio in Storybook
- For testing with real audio, replace with actual file URLs or data URLs
- Example data URL for short beep:
  ```
  data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=
  ```

## Layout Options

Stories use different layout modes:

- `fullscreen` - For VoiceRefinement main component
- `padded` - For all sub-components (default)
- `centered` - Not used in these stories

## Extending Stories

To add a new story:

1. Identify the component state you want to test
2. Create appropriate mock data
3. Add the story export
4. Include relevant parameters (viewport, background)
5. Add descriptive JSDoc comment
6. Test in Storybook

Example:
```typescript
/**
 * Voice analyzer in regenerating state with user feedback
 */
export const RegeneratingWithCustomFeedback: Story = {
  args: {
    selectedSegment: mockSegment,
    voiceInstructions: mockInstructions,
    voiceInstructionsData: mockInstructionsData,
    isAnalyzing: true,
    userFeedback: 'Make it more energetic',
    onRegenerate: async (feedback) => {
      console.log('Regenerate with feedback:', feedback);
      await new Promise(resolve => setTimeout(resolve, 2000));
    },
  },
};
```

## Troubleshooting

### Story Not Appearing
- Check that the file is named `*.stories.tsx`
- Verify the export is present and follows the pattern
- Ensure the meta object is exported as default
- Check for TypeScript errors

### Component Not Rendering
- Verify the component import path is correct
- Check that all required props are provided
- Look for console errors in Storybook
- Ensure mock data matches TypeScript interfaces

### Viewport Not Working
- Verify viewport name is correct: `mobile1`, `tablet`, or `desktop`
- Check that viewport is configured in `.storybook/preview.ts`
- Ensure parameters are at the story level, not inside args

### Background Not Working
- Verify background name: `light-gray` or `dark`
- Check that backgrounds are configured in `.storybook/preview.ts`
- Ensure parameters are at the story level, not inside args

## Best Practices

1. **Keep Stories Focused** - Each story should test one specific state or scenario
2. **Use Realistic Data** - Mock data should represent real-world scenarios
3. **Document Complex Stories** - Add JSDoc comments for clarity
4. **Test Edge Cases** - Include stories for overflow, empty states, errors
5. **Be Consistent** - Follow existing naming and structure patterns
6. **Test Responsively** - Include mobile, tablet, and desktop variants
7. **Log Interactions** - Use console.log for callback testing
8. **Avoid Side Effects** - Stories should be independent and rerunnable

## Resources

- [Storybook Documentation](https://storybook.js.org/docs)
- [Story Writing Guide](https://storybook.js.org/docs/writing-stories)
- [Component Story Format](https://storybook.js.org/docs/api/csf)
- [Storybook Args](https://storybook.js.org/docs/writing-stories/args)
- [Storybook Parameters](https://storybook.js.org/docs/writing-stories/parameters)
