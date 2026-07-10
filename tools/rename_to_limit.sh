#!/usr/bin/env bash
# One-shot rename: limit -> limit, everywhere (dirs, namespace, paths,
# env vars LIMIT_WEIGHTS/LIMIT_VERBOSE -> LIMIT_WEIGHTS/LIMIT_VERBOSE, Makefile vars).
# WORKLOG.md is left untouched as historical record.
# Must NOT run while any engine/benchmark process is alive.
set -e
cd "$(dirname "$0")/.."

git mv engine/limit engine/limit
git mv train_limit.cpp train_limit.cpp

FILES=$(grep -rl "limit\|LIMIT\|LIMIT_WEIGHTS\|LIMIT_VERBOSE" \
  --include="*.cpp" --include="*.h" --include="*.py" --include="*.sh" \
  --include="Makefile" --include=".gitignore" . 2>/dev/null | grep -v "_dev" | grep -v fusion | grep -v WORKLOG || true)
for f in $FILES; do
  sed -i 's/limit/limit/g; s/LIMIT/LIMIT/g; s/LIMIT_WEIGHTS/LIMIT_WEIGHTS/g; s/LIMIT_VERBOSE/LIMIT_VERBOSE/g' "$f"
done

# parked/merged experiment copies: all documented in WORKLOG, delete
rm -rf engine/limit_dev*
rm -f tools/uci_engine_dev*.cpp tools/uci_engine_dev*.exe

echo "rename done; files touched:"
echo "$FILES"
