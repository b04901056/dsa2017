#!/bin/bash 
g++ -std=c++11 -O2 hw2.cpp -o out
./out public_data.txt 1.in.txt result_1.txt  
diff 1.out.txt result_1.txt
./out public_data.txt 2.in.txt result_2.txt  
diff 2.out.txt result_2.txt
./out public_data.txt 3.in.txt result_3.txt  
diff 3.out.txt result_3.txt
./out public_data.txt 4.in.txt result_4.txt    
diff 4.out.txt result_4.txt