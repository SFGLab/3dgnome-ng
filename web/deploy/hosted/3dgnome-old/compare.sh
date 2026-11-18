#!/bin/bash
# GET each path from the trial copy and from the running site and say where they differ.
T=http://127.0.0.1:8776; R=http://127.0.0.1; H='Host: 3dgnome.mini.pw.edu.pl'
strip() { sed -E 's/name="csrf_token"[^>]*value="[^"]*"//g; s/value="[^"]*"([^>]*name="csrf_token")/\1/g' "$1"; }
same=0; diff=0
while read -r p; do
  [ -z "$p" ] && continue
  a=$(curl -s -o /tmp/cmp_a -w '%{http_code} %{size_download}' "$T$p"); b=$(curl -s -o /tmp/cmp_b -w '%{http_code} %{size_download}' -H "$H" "$R$p")
  if cmp -s <(strip /tmp/cmp_a) <(strip /tmp/cmp_b); then same=$((same+1)); else diff=$((diff+1)); echo "DIFF $p | trial $a | running $b | lines $(diff <(strip /tmp/cmp_a) <(strip /tmp/cmp_b) | grep -c '^[<>]')"; fi
done
echo "same $same, different $diff"
