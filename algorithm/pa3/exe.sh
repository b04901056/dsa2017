#!/bin/bash 
./router benchmarks/gr4x4.in my_result/4.out
./verify/verify_linux benchmarks/gr4x4.in my_result/4.out
./router benchmarks/gr5x5.in my_result/5.out
./verify/verify_linux benchmarks/gr5x5.in my_result/5.out
./router benchmarks/gr10x10.in my_result/10.out
./verify/verify_linux benchmarks/gr10x10.in my_result/10.out
./router benchmarks/gr20x20.in my_result/20.out
./verify/verify_linux benchmarks/gr20x20.in my_result/20.out
./router gen.in my_result/gen.out
./verify/verify_linux gen.in my_result/gen.out
#./router benchmarks/gr60x60.in my_result/60.out
#./verify/verify_linux benchmarks/gr60x60.in my_result/60.out