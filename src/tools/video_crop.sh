#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 5 ]; then
  echo "Usage: $0 <input> <w> <h> <x> <y>"
  echo "Example: $0 input.mp4 640 480 0 0"
  exit 1
fi

input="$1"
w="$2"
h="$3"
x="$4"
y="$5"

base="$(basename "$input")"
base="${base%.*}"
folder="$(dirname "$input")/$base"
mkdir -p "$folder"
name="$folder/${base}.crop.[${w},${h},${x},${y}].mp4"

ffmpeg -i "$input" -c:v libx264 -preset slow -crf 23 -c:a copy -vf "crop=${w}:${h}:${x}:${y}" "$name"
