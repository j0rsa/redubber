#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$ROOT/.git/hooks"
SOURCE_DIR="$ROOT/scripts/hooks"

for hook in "$SOURCE_DIR"/*; do
  name="$(basename "$hook")"
  target="$HOOKS_DIR/$name"
  ln -sf "$hook" "$target"
  chmod +x "$hook"
  echo "✓ Installed $name"
done

echo "All hooks installed."
