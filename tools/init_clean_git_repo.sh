#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare a clean first-release Git repository for GitHub publishing.

By default this script is a dry run. It prints the exact steps it would take
without changing any Git metadata.

Usage:
  tools/init_clean_git_repo.sh
  tools/init_clean_git_repo.sh --execute
  tools/init_clean_git_repo.sh --execute --with-internal-docs

Flags:
  --execute             Actually back up the current .git directory and re-init
                        a fresh repository on branch main.
  --with-internal-docs  Also stage AGENT.md, CLAUDE.md, and
                        CHAT_COLLECTOR_INTEGRATION_SUMMARY.md.
  --help                Show this message.
EOF
}

execute=0
with_internal_docs=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      execute=1
      ;;
    --with-internal-docs)
      with_internal_docs=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

release_paths=(
  README.md
  .gitignore
  pyproject.toml
  Makefile
  ENGINE_DEV_DOC.md
  PROFILE_SYSTEM_README.md
  docs
  examples
  pfe-cli
  pfe-core
  pfe-server
  tests
  tools
)

internal_doc_paths=(
  AGENT.md
  CLAUDE.md
  CHAT_COLLECTOR_INTEGRATION_SUMMARY.md
)

add_paths=()
for path in "${release_paths[@]}"; do
  if [[ -e "$path" ]]; then
    add_paths+=("$path")
  fi
done

if [[ "$with_internal_docs" -eq 1 ]]; then
  for path in "${internal_doc_paths[@]}"; do
    if [[ -e "$path" ]]; then
      add_paths+=("$path")
    fi
  done
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
backup_dir=".git-backup-$timestamp"

echo "Repository root: $ROOT"
echo
echo "Clean first-release plan:"
if [[ -d .git ]]; then
  echo "1. Back up current Git metadata to $backup_dir"
else
  echo "1. No existing .git directory detected; skip backup"
fi
echo "2. Initialize a fresh repository on branch main"
echo "3. Stage the curated first-release paths:"
for path in "${add_paths[@]}"; do
  echo "   - $path"
done
echo
echo "Ignored large local artifacts are controlled by .gitignore, including:"
echo "   - models/"
echo "   - trainer_job_outputs/"
echo "   - .venv/"
echo "   - .uv-cache/"
echo "   - .pip-cache/"
echo "   - tools/llama.cpp/"
echo

if [[ "$execute" -ne 1 ]]; then
  echo "Dry run only. Re-run with --execute to perform the re-init."
  exit 0
fi

if [[ -d .git ]]; then
  if [[ -e "$backup_dir" ]]; then
    echo "Backup path already exists: $backup_dir" >&2
    exit 1
  fi
  mv .git "$backup_dir"
fi

git init -b main
git add -- "${add_paths[@]}"

echo
echo "Fresh repository initialized."
echo
echo "Review staged content with:"
echo "  git status --short"
echo
echo "Suggested next steps:"
echo "  git commit -m \"Initial open-source release\""
echo "  git remote add origin <your-github-url>"
echo "  git push -u origin main"
