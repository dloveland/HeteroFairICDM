#!/bin/bash

echo Running
A=$1
B=$2
DATASET=$3

python generate_synth_data_rewire.py --a $A --b $B --dataset $DATASET
