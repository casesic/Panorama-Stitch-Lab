#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input> <width> <height>"
  echo "Example: $0 input.mp4 1280 720"
  exit 1
fi

input="$1"
w="$2"
h="$3"

base="$(basename "$input")"
base="${base%.*}"
folder="$(dirname "$input")/$base"
mkdir -p "$folder"
name="$folder/${base}.scale.[${w},${h}].mp4"

ffmpeg -i "$input" -c:v libx264 -preset slow -crf 23 -c:a copy -vf "scale=${w}:${h}" "$name"
