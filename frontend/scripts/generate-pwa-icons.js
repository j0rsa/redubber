#!/usr/bin/env node

/**
 * Generate PWA icons script
 *
 * This script generates placeholder PWA icons for development.
 * In production, replace with actual branded icons using a tool like:
 * - https://realfavicongenerator.net/
 * - https://www.pwabuilder.com/imageGenerator
 *
 * Usage:
 *   node scripts/generate-pwa-icons.js
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Simple SVG template for placeholder icons
const createSVG = (size, bgColor, textColor, text) => `
<svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
  <rect width="${size}" height="${size}" fill="${bgColor}"/>
  <text x="50%" y="50%" font-family="Arial, sans-serif" font-size="${size * 0.4}"
        fill="${textColor}" text-anchor="middle" dominant-baseline="central">
    ${text}
  </text>
</svg>
`.trim();

const publicDir = path.join(__dirname, '..', 'public');

// Ensure public directory exists
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

// Generate placeholder icons
const icons = [
  { name: 'pwa-192x192.png', size: 192 },
  { name: 'pwa-512x512.png', size: 512 },
  { name: 'apple-touch-icon.png', size: 180 },
  { name: 'favicon.ico', size: 32 }
];

console.log('Generating placeholder PWA icons...\n');

icons.forEach(({ name, size }) => {
  const svgContent = createSVG(size, '#1976d2', '#ffffff', 'R');
  const svgPath = path.join(publicDir, name.replace(/\.(png|ico)$/, '.svg'));
  const targetPath = path.join(publicDir, name);

  // Write SVG file
  fs.writeFileSync(svgPath, svgContent);
  console.log(`✓ Generated ${svgPath}`);

  // For this simple script, we'll just create SVG files
  // In production, convert to PNG/ICO using a tool like sharp or imagemagick
  if (name.endsWith('.svg')) {
    fs.copyFileSync(svgPath, targetPath);
  }
});

// Create mask-icon.svg
const maskIconSVG = `
<svg width="16" height="16" xmlns="http://www.w3.org/2000/svg">
  <path d="M2 2 L14 2 L14 14 L2 14 Z" fill="black"/>
  <text x="8" y="11" font-family="Arial" font-size="10" fill="white" text-anchor="middle">R</text>
</svg>
`.trim();

fs.writeFileSync(path.join(publicDir, 'mask-icon.svg'), maskIconSVG);
console.log(`✓ Generated ${path.join(publicDir, 'mask-icon.svg')}`);

console.log('\n✨ Placeholder PWA icons generated successfully!');
console.log('\n📝 Note: Replace these with actual branded icons before production deployment.');
console.log('   Recommended tools:');
console.log('   - https://realfavicongenerator.net/');
console.log('   - https://www.pwabuilder.com/imageGenerator\n');
