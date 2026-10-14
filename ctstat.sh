#!/bin/bash
# Overview of conformal-twin job arrays: task counts per state, one line per job.
# Usage: bash ~/ctstat.sh                  all jobs named ct-* of the last 14 days
#        bash ~/ctstat.sh 152770 152705    only these job ids (space or comma separated)
IDS=""
if [ $# -gt 0 ]; then IDS=$(echo "$*" | tr ' ' ','); SEL=(-j "$IDS"); else SEL=(-u "$USER" -S now-14days); fi
{
  sacct "${SEL[@]}" -X -n -P --format=JobID,JobName,State
  # pending array tasks are not always listed by sacct: take them from squeue, one line per task
  squeue -h -r -t PD -u "$USER" -o '%i|%j|PENDING'
} | awk -F'|' -v ids="$IDS" '
BEGIN { n = split(ids, want, ","); for (i = 1; i <= n; i++) keep[want[i]] = 1 }
{
  if (index($1, "[")) next                      # sacct range lines: pending tasks come from squeue instead
  split($1, a, "_"); id = a[1]
  if (n > 0 && !(id in keep)) next
  if (n == 0 && $2 !~ /^ct-/) next
  if (seen[$1]++) next                          # a task listed by both sacct and squeue counts once
  st = $3; sub(/ .*/, "", st); st = tolower(st)
  name[id] = $2; cnt[id SUBSEP st]++; tot[id]++; ids_[id] = 1; states[st] = 1
}
END {
  for (i in ids_) {
    line = sprintf("%-10s (%s): %3d tasks |", name[i], i, tot[i])
    for (s in states) if ((i SUBSEP s) in cnt) line = line sprintf(" %d %s", cnt[i SUBSEP s], s)
    print line
  }
}' | sort -t'(' -k2
