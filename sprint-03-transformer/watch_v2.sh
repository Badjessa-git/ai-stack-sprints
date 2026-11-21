#!/bin/bash
# One line per v2 run: latest step, latest held-out val_bpb, any crash.
cd "$(dirname "$0")"
printf "%-28s %-22s %s\n" run "last step" "last eval"
for d in runs/v2/*/; do
  n=$(basename "$d"); log="$d/train.log"
  [ -f "$log" ] || continue
  step=$(grep "^step" "$log" | tail -1 | awk '{print $2, "("$5")"}')
  ev=$(grep "eval step" "$log" | tail -1 | sed 's/eval //')
  err=$( [ -s "$d/stderr.log" ] && grep -q Error "$d/stderr.log" && echo "  !! see stderr.log")
  printf "%-28s %-22s %s%s\n" "$n" "$step" "$ev" "$err"
done
