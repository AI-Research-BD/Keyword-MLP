#!/bin/bash

ROOT=$1
curr_dir=$PWD

mkdir -p $ROOT

cd $ROOT
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O - | tar -xz

cd $curr_dir

python data_prep/helpers/make_v2_35.py -v $ROOT/validation_list.txt -t $ROOT/testing_list.txt -d $ROOT -o $ROOT