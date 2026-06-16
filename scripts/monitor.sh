#!/bin/bash
# Monitor training progress in real-time
# Usage: bash scripts/monitor.sh [project_name]

cd "$(dirname "$0")/.."
PROJECT="${1:-FIGR_SVG}"
CKPT_DIR="proj_log/$PROJECT"

echo "=== Training Monitor: $PROJECT ==="
echo ""

# Checkpoint status
echo "Checkpoints:"
ls -d "$CKPT_DIR"/step_* "$CKPT_DIR"/epoch_* 2>/dev/null | sort | while read d; do
    SIZE=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  $(basename "$d")  ($SIZE)"
done
echo ""

# GPU status
echo "GPU status:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv 2>/dev/null
echo ""

# System status
echo "System:"
echo "  Load: $(uptime | sed 's/.*load average: //')"
echo "  RAM:  $(free -h | awk '/Mem/ {print $3 "/" $2}')"
echo ""

# Latest TensorBoard events (if available)
LATEST_EVENTS=$(ls -t "$CKPT_DIR"/events.out.* 2>/dev/null | head -1)
if [ -n "$LATEST_EVENTS" ]; then
    echo "TensorBoard events: $LATEST_EVENTS"
    echo "  Run: tensorboard --logdir $CKPT_DIR --host 0.0.0.0"
fi
