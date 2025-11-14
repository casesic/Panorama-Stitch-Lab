#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input>"
  echo "Example: $0 input.mp4"
  exit 1
fi

input="$1"
base="$(basename "$input")"
base="${base%.*}"
folder="$(dirname "$input")/$base"
mkdir -p "$folder"

name0="$folder/${base}.0.mp4"
name1="$folder/${base}.1.mp4"
name2="$folder/${base}.audio.mp4"

ffmpeg -i "$input" -c:v copy -map 0:v:0 "$name0"
ffmpeg -i "$input" -c:v copy -map 0:v:1 "$name1"
ffmpeg -i "$input" -c:a copy -map 0:a:0 "$name2"
