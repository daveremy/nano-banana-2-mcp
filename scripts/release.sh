#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# release.sh — bump version, build, publish, push
# Usage: ./scripts/release.sh <major|minor|patch>
# ---------------------------------------------------------------------------

BUMP="${1:-patch}"

# Guard: must be on main with clean tree
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
  echo "ERROR: must be on main branch (currently on $BRANCH)" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is not clean" >&2
  exit 1
fi

git pull --rebase

# Run tests
echo "Running tests..."
npm test

# Bump version via npm (updates package.json and returns new version)
NEW_VERSION=$(npm version "$BUMP" --no-git-tag-version | tr -d 'v')
echo "New version: $NEW_VERSION"

# Update src/version.ts
node -e "
  const fs = require('fs');
  fs.writeFileSync('src/version.ts', 'export const VERSION = \"$NEW_VERSION\";\n');
"

# Update plugin.json (version + pinned npx package version)
node -e "
  const fs = require('fs');
  const p = '.claude-plugin/plugin.json';
  const j = JSON.parse(fs.readFileSync(p, 'utf8'));
  j.version = '$NEW_VERSION';
  j.mcpServers['nano-banana-2'].args[1] = 'nano-banana-2-mcp@$NEW_VERSION';
  fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n');
"

# Update marketplace.json
node -e "
  const fs = require('fs');
  const p = '.claude-plugin/marketplace.json';
  const j = JSON.parse(fs.readFileSync(p, 'utf8'));
  j.plugins[0].version = '$NEW_VERSION';
  fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n');
"

# Update CHANGELOG.md — replace [Unreleased] heading with new version + date
TODAY=$(date +%Y-%m-%d)
node -e "
  const fs = require('fs');
  const p = 'CHANGELOG.md';
  let c = fs.readFileSync(p, 'utf8');
  c = c.replace('## [Unreleased]', '## [Unreleased]\n\n## [$NEW_VERSION] - $TODAY');
  fs.writeFileSync(p, c);
"

# Build
echo "Building..."
npm run build

# Verify package contents
echo "Package contents:"
npm pack --dry-run

# Commit, tag, publish, push
git add -A
git commit -m "release: v$NEW_VERSION"
git tag "v$NEW_VERSION"

echo "Publishing to npm..."
npm publish

git push origin main --tags

echo ""
echo "Released v$NEW_VERSION"
