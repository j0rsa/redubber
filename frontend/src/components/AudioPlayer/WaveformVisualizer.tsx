import { useEffect, useRef } from 'react';
import styles from './WaveformVisualizer.module.css';

export interface WaveformVisualizerProps {
  isPlaying: boolean;
  progress: number; // 0-100
  className?: string;
}

/**
 * Animated waveform visualization
 * Shows animated bars that respond to playback state
 */
export const WaveformVisualizer = ({
  isPlaying,
  progress,
  className,
}: WaveformVisualizerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const barsRef = useRef<number[]>([]);

  const BAR_COUNT = 40;
  const BAR_MIN_HEIGHT = 0.1;
  const BAR_MAX_HEIGHT = 1.0;

  // Initialize bars with random heights
  useEffect(() => {
    barsRef.current = Array.from(
      { length: BAR_COUNT },
      () => Math.random() * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT) + BAR_MIN_HEIGHT
    );
  }, []);

  // Animate bars
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let targetHeights = [...barsRef.current];
    let currentHeights = [...barsRef.current];

    const animate = () => {
      const { width, height } = canvas;
      const dpr = window.devicePixelRatio || 1;

      // Set canvas size
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      const barWidth = width / BAR_COUNT;
      const gap = barWidth * 0.2;
      const actualBarWidth = barWidth - gap;

      // Update bar heights
      for (let i = 0; i < BAR_COUNT; i++) {
        // Animate towards target
        const diff = targetHeights[i] - currentHeights[i];
        currentHeights[i] += diff * 0.1;

        // Randomly change target when playing
        if (isPlaying && Math.random() < 0.05) {
          targetHeights[i] =
            Math.random() * (BAR_MAX_HEIGHT - BAR_MIN_HEIGHT) + BAR_MIN_HEIGHT;
        }

        // Calculate bar properties
        const barHeight = currentHeights[i] * height;
        const x = i * barWidth + gap / 2;
        const y = (height - barHeight) / 2;

        // Determine bar color based on progress
        const barProgress = (i / BAR_COUNT) * 100;
        const isPassed = barProgress <= progress;

        // Draw bar
        ctx.fillStyle = isPassed
          ? 'var(--color-primary)'
          : 'var(--color-border-light)';
        ctx.fillRect(x, y, actualBarWidth, barHeight);
      }

      // Continue animation
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, progress]);

  // Slow down animation when paused
  useEffect(() => {
    if (!isPlaying) {
      const targetHeights = barsRef.current.map(
        () => Math.random() * 0.3 + 0.2
      );
      barsRef.current = targetHeights;
    }
  }, [isPlaying]);

  return (
    <div className={`${styles.container} ${className || ''}`}>
      <canvas ref={canvasRef} className={styles.canvas} />
    </div>
  );
};
