#!/bin/bash
wget 'https://www.dropbox.com/s/3nlslkzoqn2sn61/model_8s_epoch_20.h5'
python test.py -m model_8s_epoch_20.h5 -d $2 -v $1
