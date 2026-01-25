#!/bin/bash
set -euo pipefail

# Configuration
GITHUB_REPO="Comfy-Org/ComfyUI"
GHCR_REGISTRY="ghcr.io/kramins/comfyui-amd"
MAX_INITIAL_VERSIONS=5

echo "🔍 Detecting ComfyUI versions to build..." >&2

# Fetch available releases from GitHub
echo "📦 Fetching ComfyUI releases from GitHub..." >&2
AVAILABLE_RELEASES=$(curl -sL "https://api.github.com/repos/${GITHUB_REPO}/releases?per_page=50" | \
    jq -r '.[].tag_name | select(startswith("v")) | ltrimstr("v")' | \
    sort -V -r)

if [ -z "$AVAILABLE_RELEASES" ]; then
    echo "❌ Error: Could not fetch releases from GitHub" >&2
    exit 1
fi

echo "📋 Found releases: $(echo "$AVAILABLE_RELEASES" | tr '\n' ', ' | sed 's/,$//')" >&2

# Fetch existing tags from GHCR
echo "🏷️  Checking existing tags in GHCR..." >&2
EXISTING_TAGS=$(curl -sL "https://ghcr.io/v2/kramins/comfyui-amd/tags/list" 2>/dev/null | \
    jq -r '.tags[]? // empty' | \
    grep -v "^latest$" | \
    grep -v "^git" | \
    sort -V -r || echo "")

if [ -z "$EXISTING_TAGS" ]; then
    echo "🆕 No existing tags found in GHCR (first run)" >&2
    FIRST_RUN=true
else
    echo "✅ Found existing tags: $(echo "$EXISTING_TAGS" | tr '\n' ', ' | sed 's/,$//')" >&2
    FIRST_RUN=false
fi

# Determine versions to build
VERSIONS_TO_BUILD=""

if [ "$FIRST_RUN" = true ]; then
    # First run: build git + last N releases
    echo "🚀 First run detected - will build 'git' + last $MAX_INITIAL_VERSIONS releases" >&2
    VERSIONS_TO_BUILD="git"
    RECENT_RELEASES=$(echo "$AVAILABLE_RELEASES" | head -n "$MAX_INITIAL_VERSIONS")
    for version in $RECENT_RELEASES; do
        VERSIONS_TO_BUILD="$VERSIONS_TO_BUILD $version"
    done
else
    # Subsequent runs: build only new releases
    echo "🔄 Checking for new releases..." >&2
    for version in $AVAILABLE_RELEASES; do
        if ! echo "$EXISTING_TAGS" | grep -q "^${version}$"; then
            echo "🆕 New version found: $version" >&2
            VERSIONS_TO_BUILD="$VERSIONS_TO_BUILD $version"
        fi
    done
    
    if [ -z "$VERSIONS_TO_BUILD" ]; then
        echo "✅ No new versions to build" >&2
    fi
fi

# Output as JSON array for GitHub Actions matrix
if [ -n "$VERSIONS_TO_BUILD" ]; then
    JSON_OUTPUT=$(echo "$VERSIONS_TO_BUILD" | tr ' ' '\n' | jq -R . | jq -cs .)
    echo "$JSON_OUTPUT"
    echo "📤 Versions to build: $VERSIONS_TO_BUILD" >&2
else
    echo "[]"
fi
