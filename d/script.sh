#!/usr/bin/bash
# Uses "regular" mathematical functions
ldc2 script.d kernel.d math.d arrays.d --release -O --d-version=verbose --boundscheck=off --mcpu=native
/usr/bin/time -v ./script
# Uses Fast Math
ldc2 script.d kernel.d math.d arrays.d --release -O --d-version=verbose --d-version=fastmath --ffast-math --boundscheck=off --mcpu=native
/usr/bin/time -v ./script
