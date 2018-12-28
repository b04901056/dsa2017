#!/bin/bash

if [ -d MFCC/ ]; then
	echo "rm MFCC/ -r"
	rm MFCC/ -r
	echo "mkdir MFCC"
	mkdir MFCC
fi

if [ -d hmm/ ]; then
	echo "rm hmm/ -r"
	rm hmm/ -r
	echo "mkdir hmm"
	mkdir hmm
fi

if [ -d result/ ]; then
	echo "rm result/ -r"
	rm result/ -r
	echo "mkdir result"
	mkdir result
fi

cd bin; make clean; cd ..

####################################################################################################################################

source set_htk_path.sh;

feature_dir=MFCC

[ -d $feature_dir/training ] || mkdir -p $feature_dir/training;
[ -d $feature_dir/testing ]  || mkdir -p $feature_dir/testing;

config=lib/hcopy.cfg
training_list=scripts/training_hcopy.scp
testing_list=scripts/testing_hcopy.scp

HCopy -T 1 -C $config -S $training_list
HCopy -T 1 -C $config -S $testing_list

####################################################################################################################################
 

source set_htk_path.sh;

config=lib/config.cfg
proto=lib/proto

# set output path and file of HCompV
init_mmf=hmmdef
mmf_dir=hmm

data_list=scripts/training.scp

HCompV -T 2 -D -C $config -o $init_mmf -f 0.01 \
	-m -S $data_list -M $mmf_dir $proto


############################################
# if you cannot run these binary file,     #
# use a c compiler to re-complie them.     #
# source codes are in the directory: bin/  #
############################################
out_macro=hmm/macros
out_model=hmm/models

if [ ! -e bin/macro ]; then
	cd bin/; make; cd ..
fi

if [ ! -e bin/models_1mixsil ]; then
	cd bin/; make; cd ..
fi

bin/macro 39 MFCC_Z_E_D_A $mmf_dir/vFloors $out_macro
bin/models_1mixsil $mmf_dir/$init_mmf $out_model

####################################################################################################################################
 

source set_htk_path.sh

config=lib/config.cfg
data_list=scripts/training.scp

mmf_dir=hmm
macro=$mmf_dir/macros
model=$mmf_dir/models

label=labels/Clean08TR.mlf
model_list=lib/models.lst

#################################################
# re-adjust mean, var
echo "step 01 [HErest]: adjust mean, var..."
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18;
do
	echo "iteration $i"
	HERest -C $config -I $label \
		-t 250.0 150.0 1000.0 -S $data_list \
		-H $macro -H $model -M $mmf_dir $model_list
done

#################################################
# add short pause model, change model_list and label file
echo "step 02 [HHEd]: add \"sp\" model"

if [ ! -e bin/spmodel_gen ]; then
	cd bin/; make ; cd ..;
fi
bin/spmodel_gen $model $model
label=labels/Clean08TR_sp.mlf
model_list=lib/models_sp.lst
HHEd -T 2 -H $macro -H $model -M $mmf_dir lib/sil1.hed $model_list

# re-adjust mean, var
echo "step 03 [HErest]: adjust mean, var..."
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18;
do
	echo "iteration $i"
	HERest -C $config -I $label \
		-t 250.0 150.0 1000.0 -S $data_list \
		-H $macro -H $model -M $mmf_dir $model_list
done


#################################################
# increase mixture
echo "step 04 [HHEd]: split gaussian mixture..."
HHEd -T 2 -H $macro -H $model -M $mmf_dir lib/mix2_10.hed $model_list

# re-adjust mean, var
echo "step 05 [HERest]: adjust mean, var..."
for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18;
do
	HERest -C $config -I $label \
		-t 250.0 150.0 1000.0 -S $data_list \
		-H $macro -H $model -M $mmf_dir $model_list
done
#############################################################################################################################################

source set_htk_path.sh

config=lib/config.cfg
macro=hmm/macros
model=hmm/models

test_data_list=scripts/testing.scp
dictionary=lib/dict

out_mlf=result/result.mlf

out_acc=result/accuracy
answer_mlf=labels/answer.mlf

model_list=lib/models_sp.lst
word_net=lib/wdnet_sp

HVite -D -H $macro -H $model -S $test_data_list -C $config -w $word_net \
	-l '*' -i $out_mlf -p 0.0 -s 0.0 $dictionary $model_list
	
HResults -e "???" sil -e "???" sp -I $answer_mlf $model_list \
	$out_mlf >> $out_acc

