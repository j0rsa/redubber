#!/bin/bash
# PWA Verification Script
# Checks that all PWA components are correctly implemented

echo "🔍 Verifying PWA Implementation..."
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

checks_passed=0
checks_failed=0

# Function to check file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((checks_passed++))
    else
        echo -e "${RED}✗${NC} $2 (missing: $1)"
        ((checks_failed++))
    fi
}

# Function to check directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((checks_passed++))
    else
        echo -e "${RED}✗${NC} $2 (missing: $1)"
        ((checks_failed++))
    fi
}

# Function to check file contains string
check_content() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $3"
        ((checks_passed++))
    else
        echo -e "${RED}✗${NC} $3 (not found in $1)"
        ((checks_failed++))
    fi
}

echo "📁 Checking Source Files..."
check_file "src/serviceWorkerRegistration.ts" "Service worker registration"
check_file "src/hooks/useNotifications.ts" "Notifications hook"
check_file "src/hooks/useOnlineStatus.ts" "Online status hook"
check_file "src/components/InstallPrompt.tsx" "Install prompt component"
check_file "src/components/OfflineBanner.tsx" "Offline banner component"
check_file "public/offline.html" "Offline fallback page"
echo ""

echo "🎨 Checking Icons..."
check_file "public/pwa-192x192.png" "192x192 icon"
check_file "public/pwa-512x512.png" "512x512 icon"
check_file "public/apple-touch-icon.png" "Apple touch icon"
check_file "public/favicon.ico" "Favicon"
check_file "public/mask-icon.svg" "Safari mask icon"
echo ""

echo "⚙️ Checking Configuration..."
check_content "vite.config.ts" "VitePWA" "VitePWA plugin imported"
check_content "vite.config.ts" "registerType.*autoUpdate" "Auto-update configured"
check_content "vite.config.ts" "NetworkFirst" "API caching strategy"
check_content "package.json" "vite-plugin-pwa" "PWA plugin dependency"
check_content "package.json" "workbox-window" "Workbox dependency"
echo ""

echo "🔗 Checking Integration..."
check_content "src/main.tsx" "registerServiceWorker" "SW registration in main"
check_content "src/main.tsx" "OfflineBanner" "Offline banner in app"
check_content "src/pages/ProjectHub.tsx" "InstallPrompt" "Install prompt in hub"
check_content "src/pages/JobMonitor.tsx" "useNotifications" "Notifications in job monitor"
echo ""

echo "📦 Checking Build Output..."
if [ -d "dist" ]; then
    check_file "dist/sw.js" "Service worker built"
    check_file "dist/manifest.webmanifest" "Manifest built"
    check_file "dist/registerSW.js" "SW registration script"
    check_file "dist/offline.html" "Offline page in dist"

    # Check manifest content
    if [ -f "dist/manifest.webmanifest" ]; then
        check_content "dist/manifest.webmanifest" "Redubber" "App name in manifest"
        check_content "dist/manifest.webmanifest" "standalone" "Standalone display mode"
        check_content "dist/manifest.webmanifest" "192x192" "Icons in manifest"
    fi
else
    echo -e "${YELLOW}⚠${NC} Build directory not found. Run 'npm run build' first."
    echo ""
fi
echo ""

echo "📚 Checking Documentation..."
check_file "PWA_TESTING.md" "Testing guide"
check_file "PWA_IMPLEMENTATION_SUMMARY.md" "Implementation summary"
check_file "PWA_QUICK_START.md" "Quick start guide"
check_file "scripts/generate-pwa-icons.js" "Icon generation script"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Summary:"
echo -e "${GREEN}✓ Passed:${NC} $checks_passed"
if [ $checks_failed -gt 0 ]; then
    echo -e "${RED}✗ Failed:${NC} $checks_failed"
    echo ""
    echo "Some checks failed. Please review the output above."
    exit 1
else
    echo -e "${GREEN}✗ Failed:${NC} 0"
    echo ""
    echo -e "${GREEN}🎉 All PWA components verified successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run 'npm run build && npm run preview'"
    echo "2. Open http://localhost:4173"
    echo "3. Test PWA features (see PWA_TESTING.md)"
    echo ""
    exit 0
fi
