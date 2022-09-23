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
from schemes import Simulation

class Full_A100_Sim(Simulation):

    def __init__(self, args):
        super().__init__(args)

    def try_schedule(self, job, gpu_list, migration, run_log):
        sched_done = False

        avail_gpu = [g for g in gpu_list if g.jobs == ['idle']]
        if len(avail_gpu) > 0:
            gpu = avail_gpu[0]
            gpu.jobs = [job]
            gpu.max_allowed = 'full'
            sched_done = True
            print(f'Schedule time: {self.Tnow}', file=run_log, flush=True)
            print(f'job {job} scheduled on GPU {gpu.index}', file=run_log, flush=True)
        return sched_done

    def run(self, args):
        # GPU status class: records current partition, and jobs running on the partition
        gpu_states = []
        for i in range(args.num_gpu):
            gpu_states.append(GPU_status(i))

        queue = list(self.queue_dict)
        queue_ind = 0
    
        # start simulation
        # ignore checkpointing overhead in simulation
        arrived_jobs = []
        active_jobs_per_gpu = [] # time series of total number of jobs running
        progress = []
        remain_time = copy.deepcopy(self.job_runtime)
    
        completion, migration = {}, {}
        for j in remain_time:
            completion[j] = 0
            migration[j] = 0
    
        run_log = open('logs/simulation_full.log','w')
        
        while True:
            # loop for getting jobs from the queue
            while queue_ind < len(queue) and self.queue_dict[queue[queue_ind]] <= self.Tnow:
                arrived_jobs.append(queue[queue_ind])
                queue_ind += 1
                
#            if self.Tnow >= 120:
#                pdb.set_trace()
            if len(arrived_jobs) >= 1:

                '''
                priority
                1. there is idle GPU, so no migration at all
                2. least number of current jobs currently running, randomly pick one, order does not matter
                '''
                for job in arrived_jobs[:]:
                    sched_done = self.try_schedule(job, gpu_states, migration, run_log)
                    if sched_done:
                        arrived_jobs.pop(0)
                        self.sched_time[job] = self.Tnow
                    else: # stop scheduling jobs, follow a strict FIFO pattern
                        break
                        
           
            ############### wait for next iteration, job is running ##########
        
            if self.Tnow % 60 == 0:
                progress.append(sum(list(completion.values())))
        
            self.Tnow += args.step
            cnt_active = 0
            # for each job currently running, reduce its remaining time proportionally
            emptied_gpu = []
            for gpu in gpu_states:
                for ind, job in enumerate(gpu.active_jobs):
                    passed_time = args.step 
                    if remain_time[job] - passed_time <= 0:
                        # job has finished, update GPU
                        completion[job] = 1
                        gpu.jobs[ind] = 'idle'
                        self.comp_time[job] = self.Tnow
                        remain_time[job] = 0
                        emptied_gpu.append(gpu)
                        gpu.max_allowed = '7g.40gb'
                        print(f'Finish time: {self.Tnow}', file=run_log, flush=True)
                        print(f'job {job} finished', file=run_log, flush=True)
                    else:
                        completion[job] += passed_time / self.job_runtime[job]
                        remain_time[job] -= passed_time

#            # first see if jobs in arrived_jobs can be scheduled on emptied gpus
            for job in arrived_jobs[:]:
                sched_done = self.try_schedule(job, emptied_gpu, migration, run_log)
                if sched_done:
                    arrived_jobs.pop(0)
                    self.sched_time[job] = self.Tnow
                else: # stop scheduling jobs, follow a strict FIFO pattern
                    break
#            if no more arrived jobs can schedule, repartition emptied gpus:
            for gpu in gpu_states:
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in gpu_states]))
        
            # sanity check
            for gpu in gpu_states:
                if 'idle' in gpu.jobs and gpu.max_allowed != '7g.40gb':
                    raise RuntimeError('Check failed: GPU should not have bubble')
        
            ################ check if termination condition is met ################
        
            if sum(completion.values()) == len(completion) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {self.Tnow}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = self.Tnow 
                self.overall_rate.append(self.span_time)
                break
        
        ########################
        Path('logs/full').mkdir(parents=True, exist_ok=True)
        JCT, JRT, QT = {}, {}, {}
        
        for job in self.job_runtime:
            JCT[job] = self.comp_time[job] - self.queue_dict[job]
            QT[job] = self.sched_time[job] - self.queue_dict[job]
            JRT[job] = self.comp_time[job] - self.sched_time[job]
        
        relative_jrt = {}
        for job in JRT:
            relative_jrt[job] = JRT[job] / self.job_runtime[job]        
        
        for metric, name in zip([JCT, JRT, QT, relative_jrt], ['JCT', 'JRT', 'QT', 'relative_jrt']):
            metric['average'] = np.mean(list(metric.values()))
            with open(f'logs/full/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/full/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/full/remain_time.json', 'w') as f:
            json.dump(remain_time, f, indent=4)
        with open('logs/full/completion.json', 'w') as f:
            json.dump(completion, f, indent=4)
        with open('logs/full/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/full/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/full/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)


