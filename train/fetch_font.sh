#!/bin/sh
# Downloads the Gurney-7B outlines (WTFPL): a tracing of the ISO 7811
# embossing digits used to render synthetic training data.
set -e
cd "$(dirname "$0")"
mkdir -p fonts/gurney/eps
base=https://raw.githubusercontent.com/scruss/Gurney-7B/master
curl -sfo fonts/gurney/COPYING "$base/COPYING"
for d in zero one two three four five six seven eight nine; do
  curl -sfo "fonts/gurney/eps/$d.eps" "$base/eps/$d.eps"
done
echo "Gurney-7B glyphs in $(pwd)/fonts/gurney/eps"
