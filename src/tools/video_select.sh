#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input> <start_frame> <end_frame>"
  echo "Example: $0 input.mp4 100 200"
  exit 1
fi

input="$1"
start="$2"
end="$3"

base="$(basename "$input")"
base="${base%.*}"
folder="$(dirname "$input")/$base"
mkdir -p "$folder"
name="$folder/${base}.select.${start}-${end}.mp4"

# Extract frames between start and end (inclusive)
ffmpeg -i "$input" -c:v libx264 -preset slow -crf 23 -c:a copy -vf "select=between(n\,$start\,$end)" -vsync 0 "$name"
