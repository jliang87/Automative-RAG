#!/bin/bash

# Usage: ./print_tree_limited.sh /path/to/your/dir

if [ -z "$1" ]; then
  echo "Usage: $0 /path/to/dir"
  exit 1
fi

TARGET_DIR="$1"

echo "📁 $TARGET_DIR"

# 1️⃣ Show all files & folders at root level
find "$TARGET_DIR" -maxdepth 1 -mindepth 1 | while read item; do
  name=$(basename "$item")
  if [ -d "$item" ]; then
    echo "|-- $name/"
  else
    echo "|-- $name"
  fi
done

# 2️⃣ Helper for pretty-printing a subfolder tree
print_subfolder() {
  local subfolder="$1"
  local full_path="$TARGET_DIR/$subfolder"

  if [ -d "$full_path" ]; then
    echo ""
    echo "|-- $subfolder/"
    find "$full_path" -mindepth 1 | sed -e "s|^$TARGET_DIR/||" -e 's|[^/]*/|   |g' -e 's|/|/|g' -e 's|^|   |'
  fi
}

# 3️⃣ Show full tree for src/ and scripts/ only
print_subfolder "src"
print_subfolder "scripts"
