#!/bin/bash

set -x # echo on

###### this is for eval sim 1 and sim 2 ######
for MODE in "miso" "full" "static" "oracle"
do
    python eval_run.py --mode $MODE --shuffle_jobs
    echo $MODE finished
done

##### this is for sweep of interarrival time ######
#for MODE in "miso" "full"
#do
#    for ARR in {1..30}
#    do        
#        python eval_run.py --mode $MODE --shuffle_jobs --arrival $ARR --sensitivity arr${ARR}        
#        echo $MODE finished
#    done
#done

##### this is for checkpoint overhead  ######
#for MODE in "miso" "static" "oracle"
#do
#    for OVHD in 10 0
#    do        
#        python eval_run.py --mode $MODE --shuffle_jobs --overhead $OVHD --sensitivity ovhd${OVHD}   
#        echo $MODE finished
#    done
#done

##### this is for prediction error  ######
#for MODE in "miso"
#do
#    python eval_run.py --mode $MODE --shuffle_jobs --rand_gen --sensitivity rand_gen &&   
#    echo $MODE finished
#    python eval_run.py --mode $MODE --shuffle_jobs --error_mean 0.032 --error_std 0.0064 --sensitivity 2x_error &&   
#    echo $MODE finished
#    python eval_run.py --mode $MODE --shuffle_jobs --error_mean 0.08 --error_std 0.016 --sensitivity 10x_error &&   
#    echo $MODE finished
#done

##### this is for MPS time  ######
#for MODE in "miso"
#do
#    python eval_run.py --mode $MODE --shuffle_jobs --mps_time 60  --sensitivity mps_time_60 &&   
#    echo $MODE finished
#    python eval_run.py --mode $MODE --shuffle_jobs --mps_rate 4 --sensitivity mps_rate_4 &&   
#    echo $MODE finished
#done

