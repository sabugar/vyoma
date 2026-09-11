#!/bin/bash
echo ">>> 15 second monitor - button 2-3 baar daba ke chhodiye <<<"
timeout 15 gpiomon --num-events=12 gpiochip0 144 2>&1 | head -14
echo "--- done ---"
