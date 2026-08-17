#!/usr/bin/env bash
# Pull the LM7VALIDATOR verdict out of a completed AWS Device Farm run.
#
#   ./fetch_devicefarm_result.sh <run-arn> [region]
#
# The harness logs its result through NSLog, so the evidence lives in the run's
# device log rather than in the run result -- a BUILTIN_FUZZ run reports PASSED
# whenever the app did not crash, which says nothing about the diff.
set -euo pipefail

RUN=${1:?usage: fetch_devicefarm_result.sh <run-arn> [region]}
REGION=${2:-us-west-2}

aws devicefarm get-run --region "$REGION" --arn "$RUN" \
    --query 'run.{status:status,result:result,deviceMinutes:deviceMinutes.total}' \
    --output table

STATUS=$(aws devicefarm get-run --region "$REGION" --arn "$RUN" --query 'run.status' --output text)
if [ "$STATUS" != "COMPLETED" ]; then
    echo "run is $STATUS; device logs appear once it is COMPLETED" >&2
    exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Device logs hang off the run's jobs, so walk them rather than the run arn.
aws devicefarm list-jobs --region "$REGION" --arn "$RUN" --query 'jobs[].arn' --output text |
    tr '\t' '\n' | while read -r job; do
    [ -n "$job" ] || continue
    aws devicefarm list-artifacts --region "$REGION" --arn "$job" --type FILE \
        --query "artifacts[?type=='DEVICE_LOG'].url" --output text |
        tr '\t' '\n' | while read -r url; do
        [ -n "$url" ] || continue
        curl -s "$url" -o "$TMP/log.txt"
        echo
        echo "--- LM7VALIDATOR lines ---"
        grep -a LM7VALIDATOR "$TMP/log.txt" || echo "(none found -- app may have crashed before logging)"
        echo "--- end ---"
    done
done
