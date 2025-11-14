#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <input> <start> <step> <end>"
  echo "Example: $0 input.mp4 0 10 100"
  exit 1
fi

input="$1"
start="$2"
step="$3"
end="$4"

# seq is used to generate the frame numbers
for i in $(seq "$start" "$step" "$end"); do
  "$(dirname "$0")/video_snap.sh" "$input" "$i"
done
