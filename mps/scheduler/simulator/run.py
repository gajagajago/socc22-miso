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
from schemes import *
from scheme_full import Full_A100_Sim
from scheme_static import Static_Sim
from scheme_oracle import Oracle_Sim
from scheme_mps import MPS_Sim

parser = argparse.ArgumentParser(description='simulator')
parser.add_argument('--arrival', type=int, help='inter-arrival period', default=60)
parser.add_argument('--num_job', type=int, help='total number of jobs', default=100)
parser.add_argument('--num_gpu', type=int, help='total number of GPUs', default=8)
parser.add_argument('--overhead', type=int, help='average migration overhead', default=5)
parser.add_argument('--mps_time', type=int, help='average MPS profiling time', default=30)
parser.add_argument('--seed', type=int, help='random seed', default=1)
parser.add_argument('--error_mean', type=float, help='mean error of predictor', default=0.016)
parser.add_argument('--error_std', type=float, help='error variance when using Gaussian to generate prediction error', default=0.0016*2)
parser.add_argument('--test', action='store_true', help='test mode', default=False)
parser.add_argument('--step', type=int, help='simulation step size', default=1)
parser.add_argument('--filler', action='store_true', help='make first 5% of jobs as filler jobs', default=False)
parser.add_argument('--flat_arrival', action='store_true', help='do not make first 50 jobs arrive more frequently', default=False)
parser.add_argument('--shuffle_jobs', action='store_true', help='shuffle job mix', default=False)
parser.add_argument('--exp_trace', action='store_true', help='use A100 collected trace', default=False)
parser.add_argument('--exp_ovhd', action='store_true', help='use collected trace overhead', default=False)
parser.add_argument('--analyze', action='store_true', help='use this switch to analyze performance on single GPU, all jobs arrive at same time ', default=False)

args = parser.parse_args()

if args.test:
    sim = Simulation(args)
    gpu_list = [GPU_status(0), GPU_status(1)]
    gpu_list[0].implement(1, [4,2,1])
    
    gpu_list[0].single_gpu_optimize(10, sim.perf_actual)
    gpu_list[0].single_gpu_optimize(1, sim.perf_actual)
    gpu_list[0].single_gpu_optimize(3, sim.perf_actual)
    gpu_list[0].single_gpu_optimize(2, sim.perf_actual)
    pdb.set_trace()

Path('logs/miso').mkdir(parents=True, exist_ok=True)
Path('logs/full').mkdir(parents=True, exist_ok=True)
Path('logs/static').mkdir(parents=True, exist_ok=True)
Path('logs/oracle').mkdir(parents=True, exist_ok=True)
Path('logs/mps').mkdir(parents=True, exist_ok=True)

miso_sim = Simulation(args)
miso_sim.run(args)
##
full_sim = Full_A100_Sim(args)
full_sim.run(args)
##
static_sim = Static_Sim(args)
static_sim.run(args, slice_code=6, partition=[3,2,2])
##
oracle_sim = Oracle_Sim(args)
oracle_sim.run(args)
##
mps_sim = MPS_Sim(args)
mps_sim.run(args)
