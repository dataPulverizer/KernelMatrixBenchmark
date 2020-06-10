#!/usr/bin/bash
# Uses "regular" mathematical functions
chpl --fast --ieee-float script.chpl kernel.chpl
/usr/bin/time -v ./script --verbose=true --fastmath=false
# Uses Fast Math
chpl --fast --no-ieee-float script.chpl kernel.chpl
/usr/bin/time -v ./script --verbose=true --fastmath=true
