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

user = os.environ['USER']

class mGPU_status(GPU_status):
    def __init__(self, args):
        super().__init__(args)
        self.multigpu = False

class miso_mgpu(Simulation):
    def __init__(self, args):
        super().__init__(args)

    def run(self, args):
        # GPU status class: records current partition, and jobs running on the partition
        gpu_states = []
        for i in range(args.num_gpu):
            gpu_states.append(mGPU_status(i))
        for gpu in gpu_states:
            self.start_mps[gpu] = 0

        queue = list(self.queue_dict)
        queue_ind = 0
    
        # start simulation
        # ignore checkpointing overhead in simulation
        arrived_jobs = []
        active_jobs_per_gpu = [] # time series of total number of jobs running
        progress = []
        remain_time = copy.deepcopy(self.job_runtime)
        runtimes = []
    
        completion, migration = {}, {}
        for j in remain_time:
            if j not in self.filler_jobs:
                completion[j] = 0
            migration[j] = 0
    
        run_log = open(f'/scratch/{user}/miso_logs/paper_eval/mgpu{args.seed}.log','w')



   
