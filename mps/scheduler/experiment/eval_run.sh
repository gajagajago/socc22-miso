#!/bin/bash

set -x # echo on

#for NUM in {2..7}
#do
#    python eval_run.py --num_job $NUM --analyze --num_gpu 1 --exp_ovhd --flat_arrival &&   
#    echo jobs $NUM finished
#done
# python eval_run.py --num_gpu 8 --exp_ovhd --error_mean 0.02 --seed $1 --shuffle_jobs 
python eval_run.py --num_gpu 8 --exp_ovhd --seed $1 --mps_time 60 --error_mean 0.016
# seed = 97