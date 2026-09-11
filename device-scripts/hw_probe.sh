#!/bin/bash
echo "=== module EEPROM ==="
i2ctransfer -f -y 0 w1@0x50 0x14 r22@0x50 2>/dev/null | tr -d ' ' | sed 's/0x//g' | xxd -r -p 2>/dev/null; echo
echo "=== carrier EEPROM ==="
i2ctransfer -f -y 0 w1@0x57 0x14 r22@0x57 2>/dev/null | tr -d ' ' | sed 's/0x//g' | xxd -r -p 2>/dev/null; echo
echo "=== gpiochip0 line 144 ==="
gpioinfo gpiochip0 2>/dev/null | sed -n '1p' ; gpioinfo 2>/dev/null | grep -i "PAC.06"
