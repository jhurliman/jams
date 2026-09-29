#!/bin/bash
# K10 runner: acquire beatport_key -> CQT features -> 5-fold CV train. Fail-fast + S3 sync.
set -uo pipefail
source ~/.aws-env; export AWS_DEFAULT_REGION=us-west-2
S3=s3://jams-mir-eval-usw2/k10
AWS=~/bootvenv/bin/aws
export PATH="$HOME/.local/bin:$PATH"

fail() { echo "FAILED: $1" >> ~/k10.log; $AWS s3 cp ~/k10.log $S3/FAILED.log; exit 1; }

echo "=== acquire $(date -u) ===" >> ~/k10.log
uv run ~/train_key_cnn.py acquire --data-home ~/bpkey >> ~/k10.log 2>&1 || fail acquire
$AWS s3 cp ~/k10.log $S3/k10.log --only-show-errors

echo "=== features $(date -u) ===" >> ~/k10.log
uv run ~/train_key_cnn.py features --data-home ~/bpkey >> ~/k10.log 2>&1 || fail features
$AWS s3 cp ~/k10.log $S3/k10.log --only-show-errors

( while true; do sleep 600; $AWS s3 sync ~/k10out $S3/out --only-show-errors 2>/dev/null; $AWS s3 cp ~/k10.log $S3/k10.log --only-show-errors 2>/dev/null; done ) &
SYNC_PID=$!

echo "=== train $(date -u) ===" >> ~/k10.log
uv run ~/train_key_cnn.py train --data-home ~/bpkey --out ~/k10out --workers 12 >> ~/k10.log 2>&1 || fail train
kill $SYNC_PID 2>/dev/null || true

$AWS s3 sync ~/k10out $S3/out --only-show-errors
$AWS s3 cp ~/k10.log $S3/k10.log --only-show-errors
touch ~/DONE; $AWS s3 cp ~/DONE $S3/DONE
echo "=== DONE $(date -u) ===" >> ~/k10.log
