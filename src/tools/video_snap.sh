#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input> <frame_number>"
  echo "Example: $0 input.mp4 42"
  exit 1
fi

input="$1"
frame="$2"

base="$(basename "$input")"
base="${base%.*}"
folder="data/$base"
mkdir -p "$folder"
name="$folder/${base}.snap.${frame}.png"

# Grab a single frame (frame numbers start at 0)
ffmpeg -i "$input" -vf "select=eq(n\,$frame)" -vsync 0 -frames:v 1 "$name"
