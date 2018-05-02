#!/bin/bash
wget 'https://www.dropbox.com/s/lpmbgxcwcfw5nqm/model_32s_epoch_18.h5'
python test.py -m model_32s_epoch_18.h5 -d $2 -v $1
