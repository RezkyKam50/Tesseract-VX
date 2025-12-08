#!/bin/bash

generate_commit_message() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local changed_files=$(git diff --cached --name-only | wc -l)
    local from_kernel=$(uname -r)   
    local author=$(whoami)   
    local os=$(grep '^NAME=' /etc/os-release | cut -d '=' -f2 | tr -d '"')
    local branch=$(git rev-parse --abbrev-ref HEAD)

cat << EOF

chore: Auto-commit $timestamp

- Authored by   : $author
- From          : $os $from_kernel

> Summary
- Files changed : $changed_files on branch $branch
- Timestamp     : $timestamp

> Purpose
Preserve latest local state.

> Type
Scheduled commit every 1:00AM local time.

EOF
}

cd $HOME/Tesseract-VX || exit 1
git add .
 
if ! git diff-index --quiet HEAD --; then
    git commit -m "$(generate_commit_message)"
    git push --force
fi