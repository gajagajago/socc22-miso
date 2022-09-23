import pdb
import time
import os
import random
import json
import numpy as np
import glob
import argparse
import math
from pathlib import Path
import copy
from paper_eval import *
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='simulator')
parser.add_argument('--arrival', type=int, help='inter-arrival period', default=60)
parser.add_argument('--num_job', type=int, help='total number of jobs', default=100)
parser.add_argument('--num_gpu', type=int, help='total number of GPUs', default=8)
parser.add_argument('--overhead', type=int, help='average migration overhead', default=5)
parser.add_argument('--mps_time', type=int, help='average MPS profiling time', default=30)
parser.add_argument('--mps_rate', type=float, help='average MPS processing rate', default=3)
parser.add_argument('--seed', type=int, help='random seed', default=9)
parser.add_argument('--error_mean', type=float, help='mean error of predictor', default=0.016)
parser.add_argument('--error_std', type=float, help='error variance when using Gaussian to generate prediction error', default=0.0016*2)
parser.add_argument('--test', action='store_true', help='test mode', default=False)
parser.add_argument('--step', type=int, help='simulation step size', default=1)
parser.add_argument('--filler', action='store_true', help='make first 5% of jobs as filler jobs', default=False)
parser.add_argument('--flat_arrival', action='store_true', help='do not make first 50 jobs arrive more frequently', default=False)
parser.add_argument('--shuffle_jobs', action='store_true', help='shuffle job mix', default=False)
parser.add_argument('--rand_gen', action='store_true', help='generate migration decisions randomly, only for MISO', default=False)
parser.add_argument('--exp_trace', action='store_true', help='use A100 collected trace', default=False)
parser.add_argument('--exp_ovhd', action='store_true', help='use collected trace overhead (unfunctional)', default=False)
parser.add_argument('--analyze', action='store_true', help='use this switch to analyze performance on single GPU, all jobs arrive at same time ', default=False)

args = parser.parse_args()

'''
args that will change during the run:
    seed.
    set shuffle_jobs
'''

def inner_loop(mode):
    if mode == 'full':
        init = full_paper(args)
    elif mode == 'miso':
        init = miso_paper(args)
    elif mode == 'oracle':
        init = oracle_paper(args)
    elif mode == 'static':
        init = static_paper(args)
    elif mode == 'mps':
        init = mps_paper(args)
    init.run(args)

usable_cores = [0]#os.sched_getaffinity(0)
Parallel(n_jobs=len(usable_cores))(delayed(inner_loop)(mode) for mode in ['mps'])#['full', 'static', 'oracle', 'miso', 'mps'])

#for mode in ['oracle', 'static']:#['full', 'miso', 'oracle', 'static']:
#    if mode == 'full':
#        init = full_paper(args)
#    elif mode == 'miso':
#        init = miso_paper(args)
#    elif mode == 'oracle':
#        init = oracle_paper(args)
#    elif mode == 'static':
#        init = static_paper(args)
#    init.run(args)

