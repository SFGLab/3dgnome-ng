#!/bin/bash
# Keeps the redo going unattended, from a laptop that reaches the workstation and through it the
# cluster. Waits for the tools queue of the workstation, then at every round pulls the finished
# chromosomes of the cluster arms and runs the readouts for the chromosomes that are whole. Ends
# when the array of the cluster has no task waiting or running and a last round is done.
#
#   WS="ssh options and host of the workstation" EDEN=user@host:/path/checkout JOB=<array id> \
#     bash playground/rnapii/redo_night.sh
#
# The workstation session forwards the agent, which the pull needs. The readouts run in the
# screen session 3dgnome of the workstation and write ~/xu/logs/readouts.<round>.log, which ends
# with the word "round done".
set -u
WS="${WS:?the ssh options and the host of the workstation}"
EDEN="${EDEN:?user@host:/path of the checkout on the cluster}"
JOB="${JOB:?the id of the array}"
ROUNDS="${ROUNDS:-60}"
EVERY="${EVERY:-1800}"
ws() { ssh -A -o ConnectTimeout=30 -o ServerAliveInterval=30 $WS "$@" 2>/dev/null; }
until ws 'grep -q "^redo tools DONE" ~/gapfill-data/redo_tools.log'; do sleep 300; done
echo "$(date +%H:%M) tools queue done"
for round in $(seq 1 "$ROUNDS"); do
  left=$(ws "ssh -o BatchMode=yes ${EDEN%%:*} 'sacct -j $JOB -n -X -o State | grep -c -E \"PENDING|RUNNING\"'" | tail -1)
  echo "$(date +%H:%M) round $round: ${left:-?} tasks of the array waiting or running"
  ws "EDEN=$EDEN/out/xu ROOT=\$HOME/xu bash ~/xu/repo/playground/rnapii/redo_pull.sh" | tail -1
  ws "screen -S 3dgnome -X screen -t readouts bash -c 'ROOT=\$HOME/xu bash ~/xu/repo/playground/rnapii/redo_readouts.sh > ~/xu/logs/readouts.$round.log 2>&1; echo round done >> ~/xu/logs/readouts.$round.log'"
  until ws "grep -q '^round done' ~/xu/logs/readouts.$round.log"; do sleep 300; done
  ws "tail -n 3 ~/xu/logs/readouts.$round.log | head -2"
  [ "${left:-1}" = "0" ] && break
  sleep "$EVERY"
done
echo "$(date +%H:%M) redo night DONE"
