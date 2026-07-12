# PWA Testing Guide

This guide explains how to test the Progressive Web App (PWA) features of Redubber v2.0.

## Features Implemented

### 1. Service Worker & Offline Support
- Auto-updating service worker with user prompt
- NetworkFirst strategy for API calls with 5-minute cache fallback
- Offline detection with visual banner

### 2. Push Notifications
- Browser notifications for job completion/failure
- Permission request on first visit to job monitor
- Notifications include video path and status

### 3. PWA Installability
- Web app manifest with proper icons
- Install prompt after 2 visits
- Standalone display mode when installed

### 4. Caching Strategy
- Static assets cached for offline use
- API responses cached for 5 minutes
- Automatic cache cleanup

## Testing Instructions

### 1. Build the Application

```bash
cd /Users/abochev/code/redubber/frontend
npm run build
npm run preview
```

### 2. Test Service Worker Registration

1. Open DevTools (F12) → Application → Service Workers
2. You should see the service worker registered for `http://localhost:4173`
3. Verify status shows "activated and is running"

### 3. Test Offline Mode

**Method 1: DevTools**
1. Open DevTools → Network tab
2. Check "Offline" checkbox
3. Reload the page
4. You should see the offline banner at the top
5. Previously cached API responses should still load

**Method 2: Airplane Mode**
1. Enable airplane mode on your device
2. Reload the app
3. Verify offline banner appears

### 4. Test PWA Installation

**Desktop (Chrome/Edge):**
1. Visit the app 2 times (to trigger the prompt)
2. You should see an install prompt at the bottom
3. Click "Install" button
4. The app opens in a standalone window
5. Check Applications folder (Mac) or Start Menu (Windows)

**Desktop Manual:**
1. Look for install icon in address bar (⊕ or computer icon)
2. Click it and select "Install Redubber"

**Mobile (Chrome/Safari):**
1. Chrome: Tap three dots → "Add to Home screen"
2. Safari: Tap Share button → "Add to Home Screen"
3. Icon appears on home screen
4. Opens in fullscreen without browser UI

### 5. Test Push Notifications

**Setup:**
1. Navigate to a job monitor page (`/job/:taskId`)
2. Allow notifications when prompted
3. Or manually enable in browser settings

**Testing:**
1. Start a redubbing job
2. Navigate away or minimize the window
3. When job completes, you should receive a notification
4. Click notification to return to the job page

**Manual Testing:**
- Open DevTools → Console
- Run: `new Notification('Test', { body: 'Testing notifications', icon: '/pwa-192x192.png' })`

### 6. Test PWA Manifest

1. Open DevTools → Application → Manifest
2. Verify all fields are present:
   - Name: "Redubber - AI Video Redubbing"
   - Short name: "Redubber"
   - Start URL: "/"
   - Display: "standalone"
   - Theme color: "#1976d2"
3. Check all three icon sizes (192x192, 512x512) are loaded

### 7. Test Cache Behavior

**API Caching:**
1. Load a project page with network enabled
2. Open DevTools → Network
3. Enable offline mode
4. Reload - cached API responses should load
5. After 5 minutes, cache expires

**Static Asset Caching:**
1. Load any page
2. Open DevTools → Application → Cache Storage
3. Expand "workbox-precache-v2-..."
4. Verify JS, CSS, HTML files are cached

### 8. Test Service Worker Updates

**Simulate Update:**
1. Edit `vite.config.ts` (change version comment)
2. Rebuild: `npm run build`
3. Reload the app (with DevTools open)
4. Look for service worker "waiting" state
5. You should see a confirmation dialog: "New version available! Reload to update?"
6. Click OK to reload with new version

### 9. Lighthouse PWA Audit

```bash
# Build the app
npm run build
npm run preview

# In Chrome DevTools:
# 1. Open Lighthouse tab
# 2. Select "Progressive Web App" category
# 3. Click "Analyze page load"
```

**Expected Scores:**
- ✓ Installable
- ✓ PWA Optimized
- ✓ Service Worker registered
- ✓ Works offline
- ✓ Configured for a custom splash screen
- ✓ Sets a theme color
- ✓ Content sized correctly for viewport

### 10. Cross-Browser Testing

**Chrome/Edge (Best Support):**
- Full PWA support
- Install prompts work natively
- Push notifications supported

**Firefox:**
- Service workers work
- No install prompt (manual install only)
- Push notifications supported

**Safari (iOS/macOS):**
- Service workers supported (iOS 11.3+)
- Manual install via Share menu
- Push notifications require additional setup

## Common Issues

### Service Worker Not Registering
- Check HTTPS requirement (localhost bypasses this)
- Verify `sw.js` is being served
- Check browser console for errors

### Notifications Not Working
- Ensure HTTPS (required for notifications)
- Check browser notification permissions
- Some browsers block if notification API called too early

### Install Prompt Not Showing
- Chrome only shows after 2+ visits
- User must engage with site (click/tap)
- Won't show if already installed
- Won't show if previously dismissed

### Cache Not Working
- Check DevTools → Application → Cache Storage
- Verify service worker is active
- Check Network tab for cache hits (size column shows "from ServiceWorker")

## Production Checklist

Before deploying to production:

- [ ] Replace placeholder icons with branded 192x192, 512x512 PNG files
- [ ] Generate proper favicon.ico (multi-resolution)
- [ ] Update manifest colors to match brand
- [ ] Test on multiple devices (iOS, Android, Desktop)
- [ ] Configure HTTPS for your domain
- [ ] Set up backend VAPID keys for push notifications (future enhancement)
- [ ] Test with Lighthouse (aim for 100% PWA score)
- [ ] Verify offline fallback page is branded

## Icon Generation

Current icons are SVG placeholders. For production:

```bash
# Option 1: Use online tool
# Visit https://realfavicongenerator.net/
# Upload a 512x512 PNG logo
# Download and extract to public/

# Option 2: Use pwa-asset-generator
npm install -g pwa-asset-generator
pwa-asset-generator logo.png public/ --manifest public/manifest.json
```

## Architecture Notes

### Files Structure
```
frontend/
├── src/
│   ├── serviceWorkerRegistration.ts  # SW registration logic
│   ├── hooks/
│   │   ├── useNotifications.ts       # Push notification hook
│   │   └── useOnlineStatus.ts        # Offline detection hook
│   └── components/
│       ├── InstallPrompt.tsx         # PWA install prompt
│       └── OfflineBanner.tsx         # Offline indicator
├── public/
│   ├── offline.html                  # Offline fallback page
│   ├── pwa-192x192.png              # PWA icon
│   ├── pwa-512x512.png              # PWA icon
│   └── manifest.webmanifest          # Auto-generated by vite-plugin-pwa
└── vite.config.ts                    # PWA configuration
```

### Service Worker Strategy
- **Precaching**: Static assets (JS, CSS, HTML)
- **NetworkFirst**: API calls (fresh data preferred, cache fallback)
- **Cache expiration**: 5 minutes for API responses

### Browser Support
- Chrome 90+: Full support
- Edge 90+: Full support  
- Firefox 90+: Service workers + notifications (no install prompt)
- Safari 14+: Service workers only (limited PWA features)
- Safari iOS 11.3+: Add to home screen supported

## Resources

- [MDN PWA Guide](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
- [web.dev PWA](https://web.dev/progressive-web-apps/)
- [PWA Builder](https://www.pwabuilder.com/)
- [Workbox Documentation](https://developers.google.com/web/tools/workbox)
