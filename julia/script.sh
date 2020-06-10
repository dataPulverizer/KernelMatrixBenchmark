#!/usr/bin/bash
# Uses "regular" mathematical functions
/usr/bin/time -v julia script.jl data true
# Uses Fast Math
/usr/bin/time -v julia --math-mode=fast script.jl fmdata true
