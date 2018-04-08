#!/bin/bash
python p3_b.py -trp train-10 -tep test-100 -func hard_sum -int 100 -clu 50 -iter 5000 -cat coast 
python p3_b.py -trp train-10 -tep test-100 -func hard_sum -int 100 -clu 50 -iter 5000 -cat forest
python p3_b.py -trp train-10 -tep test-100 -func hard_sum -int 100 -clu 50 -iter 5000 -cat highway
python p3_b.py -trp train-10 -tep test-100 -func hard_sum -int 100 -clu 50 -iter 5000 -cat mountain
python p3_b.py -trp train-10 -tep test-100 -func hard_sum -int 100 -clu 50 -iter 5000 -cat suburb

python p3_b.py -trp train-10 -tep test-100 -func soft_sum -int 100 -clu 50 -iter 5000 -cat coast
python p3_b.py -trp train-10 -tep test-100 -func soft_sum -int 100 -clu 50 -iter 5000 -cat forest
python p3_b.py -trp train-10 -tep test-100 -func soft_sum -int 100 -clu 50 -iter 5000 -cat highway
python p3_b.py -trp train-10 -tep test-100 -func soft_sum -int 100 -clu 50 -iter 5000 -cat mountain
python p3_b.py -trp train-10 -tep test-100 -func soft_sum -int 100 -clu 50 -iter 5000 -cat suburb

python p3_b.py -trp train-10 -tep test-100 -func soft_max -int 100 -clu 50 -iter 5000 -cat coast
python p3_b.py -trp train-10 -tep test-100 -func soft_max -int 100 -clu 50 -iter 5000 -cat forest
python p3_b.py -trp train-10 -tep test-100 -func soft_max -int 100 -clu 50 -iter 5000 -cat highway
python p3_b.py -trp train-10 -tep test-100 -func soft_max -int 100 -clu 50 -iter 5000 -cat mountain
python p3_b.py -trp train-10 -tep test-100 -func soft_max -int 100 -clu 50 -iter 5000 -cat suburb

