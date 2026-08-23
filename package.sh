#!/bin/bash
# Bundle the submission-ready deliverable.
set -e
cd "$(dirname "$0")"
bash build.sh > /dev/null   # always package a PDF built from the current sources
rm -rf /tmp/rlvr_pkg && mkdir -p /tmp/rlvr_pkg/rlvr-universality
cp -r src paper figures results README.md NOTES.md build.sh package.sh /tmp/rlvr_pkg/rlvr-universality/
cd /tmp/rlvr_pkg/rlvr-universality
rm -rf src/__pycache__ paper/*.aux paper/*.log paper/*.out paper/*.blg paper/*.fdb_latexmk
rm -f src/queue*.sh src/prebuild.py results/theory_backup.json
# Never ship superseded runs: the quarantined 600-step GRPO sweep, the pre-clean-floor
# factorial, or the E1 seeds beyond the reported budget.  figures.py reads none of them.
rm -rf results/_quarantine
rm -f results/exp3_shard*.jsonl
cd /tmp/rlvr_pkg && zip -qr /home/claude/rlvr-selection/rlvr-universality.zip rlvr-universality
echo "packaged: $(du -h /home/claude/rlvr-selection/rlvr-universality.zip | cut -f1)"
