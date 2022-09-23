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
from utils import *
import copy
from paper_eval import *
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='simulator')
parser.add_argument('--arrival', type=int, help='inter-arrival period', default=10)
parser.add_argument('--num_job', type=int, help='total number of jobs', default=1000)
parser.add_argument('--num_gpu', type=int, help='total number of GPUs', default=40)
parser.add_argument('--overhead', type=int, help='average migration overhead', default=5)
parser.add_argument('--mps_time', type=int, help='average MPS profiling time', default=30)
parser.add_argument('--mps_rate', type=float, help='average MPS processing rate', default=3)
parser.add_argument('--seed', type=int, help='random seed', default=1)
parser.add_argument('--error_mean', type=float, help='mean error of predictor', default=0.016)
parser.add_argument('--error_std', type=float, help='error variance when using Gaussian to generate prediction error', default=0.0016*2)
parser.add_argument('--test', action='store_true', help='test mode', default=False)
parser.add_argument('--step', type=int, help='simulation step size', default=1)
parser.add_argument('--filler', action='store_true', help='make first 5% of jobs as filler jobs', default=False)
parser.add_argument('--flat_arrival', action='store_true', help='do not make first 50 jobs arrive more frequently', default=False)
parser.add_argument('--shuffle_jobs', action='store_true', help='shuffle job mix', default=False)
parser.add_argument('--mode', type=str, help='choose from miso, full, static, oracle')
parser.add_argument('--sensitivity', type=str, help='choose which sensitivity simulation to perform', default='none')
parser.add_argument('--rand_gen', action='store_true', help='generate migration decisions randomly, only for MISO', default=False)
parser.add_argument('--exp_trace', action='store_true', help='use exp trace (obsolete)', default=False)
parser.add_argument('--analyze', action='store_true', help='analyze (obsolete)', default=False)
parser.add_argument('--exp_ovhd', action='store_true', help='use experiment overhead', default=False)

args = parser.parse_args()

'''
args that will change during the run:
    seed.
    set shuffle_jobs
'''

def inner_loop(seed):
    args.seed = seed
    if args.mode == 'full':
        init = full_paper(args)
    elif args.mode == 'miso':
        init = miso_paper(args)
    elif args.mode == 'oracle':
        init = oracle_paper(args)
    elif args.mode == 'static':
        init = static_paper(args)

    return init.run(args)

usable_cores = os.sched_getaffinity(0)
results = Parallel(n_jobs=len(usable_cores))(delayed(inner_loop)(seed) for seed in range(100)) #TODO

data_jct = []
data_rate = []
data_span = []
#runtime = []

for i in results:
    data_jct.append(i[0])
    data_rate.append(i[1])
    data_span.append(i[2])
#    runtime.append(i[3])

#print(np.mean(runtime))
#print(np.max(runtime))

if args.sensitivity == 'none':
    with open(f'logs/monte_carlo/{args.mode}_jct.json', 'w') as f:
        json.dump(data_jct, f, indent=4)
    with open(f'logs/monte_carlo/{args.mode}_rate.json', 'w') as f:
        json.dump(data_rate, f, indent=4)
    with open(f'logs/monte_carlo/{args.mode}_span.json', 'w') as f:
        json.dump(data_span, f, indent=4)
else:
    Path(f'logs/monte_carlo/{args.sensitivity}').mkdir(parents=True, exist_ok=True)
    with open(f'logs/monte_carlo/{args.sensitivity}/{args.mode}_jct.json', 'w') as f:
        json.dump(data_jct, f, indent=4)
    with open(f'logs/monte_carlo/{args.sensitivity}/{args.mode}_rate.json', 'w') as f:
        json.dump(data_rate, f, indent=4)
    with open(f'logs/monte_carlo/{args.sensitivity}/{args.mode}_span.json', 'w') as f:
        json.dump(data_span, f, indent=4)

