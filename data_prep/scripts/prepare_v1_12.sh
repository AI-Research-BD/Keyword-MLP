#!/bin/bash

WORKING_DIR=$PWD
ROOT=$1


mkdir -p $ROOT/speech_commands_v1_12/
mkdir -p $ROOT/speech_commands_v1_12_test/

cd $ROOT/speech_commands_v1_12/
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O - | tar -xz

cd $WORKING_DIR
cd $ROOT/speech_commands_v1_12_test/
wget http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz -O - | tar -xz

cd $WORKING_DIR

python data_prep/helpers/make_v1_12.py --root $ROOT/speech_commands_v1_12/ --test $ROOT/speech_commands_v1_12_test/

rm -r $ROOT/speech_commands_v1_12_test/