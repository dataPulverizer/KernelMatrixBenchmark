#!/usr/bin/bash
cd d
printf "\n#============== Running D Benchmark ==============#\n"
./script.sh
cd ../chapel
printf "\n#============== Running Chapel Benchmark ==============#\n"
./script.sh
cd ../julia
printf "\n#============== Running Julia Benchmark ==============#\n"
./script.sh
printf "\n#============== Running Mir NDSlice Benchmark ==============#\n"
cd ../ndslice
dub build --compiler=ldc2 --build=release --force
/usr/bin/time -v ./ndslice
