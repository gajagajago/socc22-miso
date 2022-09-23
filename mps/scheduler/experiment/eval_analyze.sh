#!/bin/bash

set -x # echo on

for NUM in {1..10}
do
    python eval_run.py --num_job $NUM --analyze --num_gpu 1 --exp_ovhd --flat_arrival --seed $1 --shuffle_jobs 
    echo jobs $NUM finished
done

