#!/bin/bash
# Sample board power at idle, then during a full pipeline query.
echo "IDLE (5 samples):"
timeout 6 tegrastats --interval 1000 2>/dev/null | head -5 | grep -oE "VDD_IN [0-9]+mW" | awk '{s+=$2; n++} END {printf "  avg VDD_IN %.0f mW\n", s/n}'
