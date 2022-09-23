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

parser = argparse.ArgumentParser(description='simulator')
parser.add_argument('--arrival', type=int, help='inter-arrival period', default=60)
parser.add_argument('--num_job', type=int, help='total number of jobs', default=200)
parser.add_argument('--num_gpu', type=int, help='total number of GPUs', default=8)
parser.add_argument('--overhead', type=int, help='average migration overhead', default=30)
parser.add_argument('--seed', type=int, help='random seed', default=97)
parser.add_argument('--error_mean', type=float, help='mean error of predictor', default=0.016)
parser.add_argument('--error_std', type=float, help='error variance when using Gaussian to generate prediction error', default=0.0016*2)
parser.add_argument('--test', action='store_true', help='test mode', default=False)
parser.add_argument('--step', type=int, help='simulation step size', default=1)
parser.add_argument('--max_q', type=int, help='maximum number of jobs in queue', default=1)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed+1)

Username = os.environ.get('USER')

with open('../trace/trace_100.json') as f:
    job_dict = json.load(f)

job_runtime = {} # job information
for i in range(args.num_job):
    index = i % 100
    job_runtime[i] = int(job_dict[str(index)])

arrival_order = random.sample(list(job_runtime), len(job_runtime)) # job arrival order

queue_dict = {} # contains job arrive time
arrival_time = 0 
for job in arrival_order:
    arrival_time += np.random.poisson(args.arrival)
    queue_dict[job] = arrival_time

STEP_SIZE=1
Tnow = 0 # current passed time

queue_timer = Tnow
sched_time = {} # used to record queueing delay
for job in job_runtime:
    sched_time[job] = 0
comp_time = {} # used to record JCT and JRT
for job in job_runtime:
    comp_time[job] = 0

# need to determine the job runtime on different slices by randomly generating model and batch sizes
with open('../../configs/batch.json') as f:
    configs = json.load(f)
if Path('job_models.json').is_file():
    with open('job_models.json') as f:
        job_models = json.load(f)
else:
    job_models = {} # {1: 'resnet_32'}
    for job in job_dict:
        model = random.choice(list(configs))
        bs = random.choice(configs[model])
        job_models[job] = f'{model}_train{bs}'
    with open('job_models.json', 'w') as f:
        json.dump(job_models, f, indent=4)

# map job model to speedup (predicted and actual)
perf_actual, perf_pred = get_speedup(job_models, args.error_mean, args.error_std)

# GPU status class: records current partition, and jobs running on the partition
gpu_states = []
for i in range(args.num_gpu):
    gpu_states.append(GPU_status(i))

MAX_Q = 1 # maximum 7 GPUs in queue
queue = list(queue_dict)
queue_ind = 0

########### test code ###########
if args.test:
    job_list = [2,4,4,7]
    gpu_list = [GPU_status(0), GPU_status(1)]
    gpu_list[0].implement(1, [4,2,1])
    
    gpu_list[0].single_gpu_optimize(2, perf_actual)
#    gpu_list[0].single_gpu_optimize(1, perf_actual)
    gpu_list[0].single_gpu_optimize(3, perf_actual)
    gpu_list[0].single_gpu_optimize(5, perf_actual)
#    gpu_list[0].single_gpu_optimize(10, perf_actual)


    pdb.set_trace()

    gpu_list[0].jobs[0] = 'idle'
    gpu_list[0].jobs[1] = 'idle'
    pdb.set_trace()
    num_mig = gpu_list[0].idle_partition_optimize(perf_actual)

remain_time = copy.deepcopy(job_runtime)
completion, migration = {}, {}
for j in remain_time:
    completion[j] = 0
    migration[j] = 0
run_log = open('simulation.log','w')

# start simulation
# ignore checkpointing overhead in simulation
arrived_jobs = []
active_jobs_per_gpu = [] # time series of total number of jobs running
progress = []
while True:
    # if no jobs to schedule, look from the queue
    if len(arrived_jobs) == 0:
        for i in range(MAX_Q):
            if queue_ind < len(queue) and queue_dict[queue[queue_ind]] <= Tnow:
                arrived_jobs.append(queue[queue_ind])
                queue_ind += 1
            else:
                break
    if len(arrived_jobs) == 1:
        '''
        priority
        1. there is idle GPU, so no migration at all
        2. least number of current jobs currently running, randomly pick one, order does not matter
        '''
        sched_done = False
        job = arrived_jobs[0]
        for instance in GPU_status.n2s_reverse.values():
            if instance in perf_pred[job%99]:
                min_size = instance
                break
        allowed_gpus = []
        for gpu in gpu_states:
            if gpu.max_allowed != 'full':
                if int(gpu.max_allowed.split('g.')[0]) >= int(min_size.split('g.')[0]):
                    allowed_gpus.append(gpu)
        # sort by number of active jobs
        sorted_gpus = sorted(allowed_gpus, key=lambda x: len(x.active_jobs), reverse=False)
        if len(sorted_gpus) > 0:
            gpu_sched = sorted_gpus[0]
            for j in gpu_sched.active_jobs:
                migration[j] += 1
            degrade = gpu_sched.single_gpu_optimize(job, perf_pred)
            sched_done = True
            print(f'Schedule time: {Tnow}', file=run_log, flush=True)
            print(f'job {job} scheduled on GPU {gpu_sched.index}, jobs {gpu_sched.jobs}, slice {gpu_sched.partition}, new mean degradation {degrade}', file=run_log, flush=True)

        if sched_done:
            arrived_jobs = []
            sched_time[job] = Tnow
   
    ############### wait for next iteration, job is running ##########

    if Tnow % 60 == 0:
        progress.append(sum(list(completion.values())))

    Tnow += STEP_SIZE
    cnt_active = 0
    # for each job currently running, reduce its remaining time proportionally
    for gpu in gpu_states:
        slow_down = gpu.eval_degradation(perf_actual)
        for ind, job in enumerate(gpu.active_jobs):
            passed_time = STEP_SIZE / slow_down[ind]
            if remain_time[job] - passed_time <= 0:
                # job has finished, update GPU
                completion[job] = 1
                gpu.jobs[ind] = 'idle'
                comp_time[job] = Tnow
                remain_time[job] = 0
                print(f'Finish time: {Tnow}', file=run_log, flush=True)
                print(f'job {job} finished', file=run_log, flush=True)
            else:
                completion[job] += passed_time / job_runtime[job]
                remain_time[job] -= passed_time
        if 'idle' in gpu.jobs:
            num_mig, migrated_jobs = gpu.idle_partition_optimize(perf_pred)
            for j in migrated_jobs:
                migration[j] += 1
            if len(gpu.active_jobs) > 0:
                print(f'MIG re-partition on GPU {gpu.index}, jobs {gpu.jobs}, slice {gpu.partition}, {num_mig} migration counts', file=run_log, flush=True)

#        if Tnow % 60 == 0
        cnt_active += len(gpu.active_jobs)
    active_jobs_per_gpu.append(cnt_active / args.num_gpu)

    # sanity check
    for gpu in gpu_states:
        if 'idle' in gpu.jobs and gpu.max_allowed != '7g.40gb':
            raise RuntimeError('Check failed: GPU should not have bubble')

    ################ check if termination condition is met ################

    if sum(completion.values()) == len(completion) and queue_ind == args.num_job:
        print(f'Time: {Tnow}, all jobs are finished!', file=run_log, flush=True)
        break

########################
JCT, JRT, QT = {}, {}, {}

for job in job_runtime:
    JCT[job] = comp_time[job] - queue_dict[job]
    QT[job] = sched_time[job] - queue_dict[job]
    JRT[job] = comp_time[job] - sched_time[job]

relative_jrt = {}
for job in JRT:
    relative_jrt[job] = JRT[job] / job_runtime[job]        

for metric, name in zip([JCT, JRT, QT, relative_jrt], ['JCT', 'JRT', 'QT', 'relative_jrt']):
    metric['average'] = np.mean(list(metric.values()))
    with open(f'logs/miso/{name}.json', 'w') as f:
        json.dump(metric, f, indent=4)
migration['average'] = np.mean(list(migration.values()))

        

with open('logs/miso/active_jobs_per_gpu.json', 'w') as f:
    json.dump(active_jobs_per_gpu, f, indent=4)
with open('logs/miso/remain_time.json', 'w') as f:
    json.dump(remain_time, f, indent=4)
with open('logs/miso/completion.json', 'w') as f:
    json.dump(completion, f, indent=4)
with open('logs/miso/progress.json', 'w') as f:
    json.dump(progress, f, indent=4)
with open('logs/miso/migration.json', 'w') as f:
    json.dump(migration, f, indent=4)


