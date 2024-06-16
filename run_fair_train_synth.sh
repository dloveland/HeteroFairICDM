#!/bin/bash

DATASET=$1
MODEL=$2
CLASS_POWER=$3
SEED=$4
NCLASS=$5
A=$6
B=$7

echo $DATASET
echo $NCLASS
python train_fair_gnns.py --dataset $DATASET --model $MODEL --nclass $NCLASS --class_power $CLASS_POWER --seed $SEED --a $A --b $B