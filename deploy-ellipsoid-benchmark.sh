#!/usr/bin/env bash
# Push the ellipsoid-benchmark code to the cluster.
# Run from the repo root. Data on the cluster is not touched.
set -euo pipefail

HOST=esul01@rci.hitec-hamburg.org
DEST=/home/esul01/hearing-ellipsoid-bench
SRC=ellipsoid-benchmark

tar czf /tmp/deploy.tgz -C "$SRC" \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
  src scripts jobs tests pyproject.toml

scp -q /tmp/deploy.tgz "$HOST:/tmp/"
ssh "$HOST" "cd $DEST && rm -rf src scripts jobs tests && \
             tar xzf /tmp/deploy.tgz && rm /tmp/deploy.tgz && \
             ls scripts jobs"
rm /tmp/deploy.tgz
echo "done."
