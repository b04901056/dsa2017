#!/bin/bash
python p3_b.py -trp train-100 -tep test-100 -func hard_sum -int 50 -clu 50 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_sum -int 50 -clu 50 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_max -int 50 -clu 50 -iter 5000

python p3_b.py -trp train-100 -tep test-100 -func hard_sum -int 50 -clu 100 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_sum -int 50 -clu 100 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_max -int 50 -clu 100 -iter 5000


python p3_b.py -trp train-100 -tep test-100 -func hard_sum -int 150 -clu 50 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_sum -int 150 -clu 50 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_max -int 150 -clu 50 -iter 5000
 
python p3_b.py -trp train-100 -tep test-100 -func hard_sum -int 150 -clu 100 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_sum -int 150 -clu 100 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_max -int 150 -clu 100 -iter 5000



python p3_b.py -trp train-100 -tep test-100 -func hard_sum -int 300 -clu 50 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_sum -int 300 -clu 50 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_max -int 300 -clu 50 -iter 5000

python p3_b.py -trp train-100 -tep test-100 -func hard_sum -int 300 -clu 100 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_sum -int 300 -clu 100 -iter 5000
python p3_b.py -trp train-100 -tep test-100 -func soft_max -int 300 -clu 100 -iter 5000
