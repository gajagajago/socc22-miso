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
user = os.environ['USER']

class MPS_Sim(Simulation):

    def __init__(self, args):
        super().__init__(args)    
        self.jrt_actual = {}

    def try_schedule(self, job, gpu_list, migration, run_log):
        sched_done = False

        avail_gpu = [g for g in gpu_list if not g.full]
        if len(avail_gpu) > 0:
            gpu = avail_gpu[0]
            gpu.jobs.append(job)
            sched_done = True
            print(f'Schedule time: {self.Tnow}', file=run_log, flush=True)
            print(f'job {job} scheduled on GPU {gpu.index}', file=run_log, flush=True)
        return sched_done

    def get_rate(self, gpu):
        deg = gpu.eval_degradation(self.jrt_actual)        
        return round(sum(deg),3)

    def run(self, args):
        # GPU status class: records current partition, and jobs running on the partition
        gpu_states = []
        for i in range(args.num_gpu):
            gpu_states.append(MPS_GPU_Status(i))

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
    
        run_log = open('logs/simulation_mps.log','w')

        # generate jrt_actual
        with open(f'/home/{user}/GIT/mig_exp/logs/full/JRT.json') as f:
            read_full = json.load(f)
        with open(f'/home/{user}/GIT/mig_exp/logs/mps/JRT.json') as f:
            read_mps = json.load(f)            
        key_list = list(read_full.keys())
        key_list.remove('average')        
        for j in remain_time:
            rand_key = random.choice(key_list)
            self.jrt_actual[j] = {'full': read_full[rand_key], 'mps': read_mps[rand_key]}

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
                slow_down = gpu.eval_degradation(self.jrt_actual)
                # TODO: continue and compare with static
                for ind, job in enumerate(gpu.active_jobs):
                    passed_time = args.step * slow_down[ind]
                    if remain_time[job] - passed_time <= 0:
                        # job has finished, update GPU
                        completion[job] = 1
                        gpu.jobs.remove(job)
                        self.comp_time[job] = self.Tnow
                        remain_time[job] = 0
                        emptied_gpu.append(gpu)
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
                if len(gpu.jobs) > 3:
                    raise RuntimeError('Check failed: GPU should not have >3 jobs')
        
            ################ check if termination condition is met ################
        
            if sum(completion.values()) == len(completion) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {self.Tnow}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = self.Tnow 
                self.overall_rate.append(self.span_time)
                break
        
        ########################
        Path('logs/mps').mkdir(parents=True, exist_ok=True)
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
            with open(f'logs/mps/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/mps/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/mps/remain_time.json', 'w') as f:
            json.dump(remain_time, f, indent=4)
        with open('logs/mps/completion.json', 'w') as f:
            json.dump(completion, f, indent=4)
        with open('logs/mps/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/mps/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/mps/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)


