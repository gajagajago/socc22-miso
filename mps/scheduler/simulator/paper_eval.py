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
user = os.environ['USER']

class full_paper(Full_A100_Sim):
    def __init__(self, args):
        super().__init__(args)
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
    
        run_log = open(f'/scratch/{user}/miso_logs/paper_eval/full{args.seed}.log','w')
        
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
        JCT = {}
        
        for job in self.job_runtime:
            if job not in self.filler_jobs:
                JCT[job] = self.comp_time[job] - self.queue_dict[job]

        return np.mean(list(JCT.values())), np.mean(self.overall_rate[:-1]), self.span_time

class static_paper(Static_Sim):
    def __init__(self, args):
        super().__init__(args)
    def run(self, args, slice_code=6, partition=[3,2,2]):
        # GPU status class: records current partition, and jobs running on the partition
        gpu_states = []
        for i in range(args.num_gpu):
            gpu_states.append(GPU_status(i))
        # static partition
        for gpu in gpu_states:
            gpu.implement(slice_code, partition)
            gpu.max_allowed = GPU_status.num_to_str[max(partition)]

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
    
        run_log = open(f'/scratch/{user}/miso_logs/paper_eval/static{args.seed}.log','w')
        
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
                        arrived_jobs.remove(job)
                        self.sched_time[job] = self.Tnow
           
            ############### wait for next iteration, job is running ##########
        
            if self.Tnow % 60 == 0:
                progress.append(sum(list(completion.values())))
        
            self.Tnow += args.step            
            cnt_active = 0
            for job in list(self.during_ovhd):
                if self.Tnow - self.start_ovhd[job] >= args.overhead:
                    self.during_ovhd.remove(job)
            # for each job currently running, reduce its remaining time proportionally
            emptied_gpu = []
            for gpu in gpu_states:
                slow_down = gpu.eval_degradation(self.perf_actual)
                for ind, job in enumerate(gpu.active_jobs):
                    if job in self.during_ovhd:
                        passed_time = 0
                    else:
                        passed_time = args.step / slow_down[ind]
                    if remain_time[job] - passed_time <= 0:
                        # job has finished, update GPU
                        completion[job] = 1
                        real_ind = gpu.jobs.index(job)
                        gpu.jobs[real_ind] = 'idle'
                        self.comp_time[job] = self.Tnow
                        remain_time[job] = 0
                        emptied_gpu.append(gpu)
                        gpu.static_max_slice()
                        print(f'Finish time: {self.Tnow}', file=run_log, flush=True)
                        print(f'job {job} finished', file=run_log, flush=True)
                    else:
                        completion[job] += passed_time / self.job_runtime[job]
                        remain_time[job] -= passed_time

#            # first see if jobs in arrived_jobs can be scheduled on emptied gpus
            for job in arrived_jobs[:]:
                sched_done = self.try_schedule(job, emptied_gpu, migration, run_log)
                if sched_done:
                    arrived_jobs.remove(job)
                    self.sched_time[job] = self.Tnow

#            if no more arrived jobs can schedule, repartition emptied gpus:
            for gpu in gpu_states:
                if 'idle' in gpu.jobs:
                    migrated_jobs = gpu.static_idle_optimize_V2()
                    gpu.update_static_migration(migrated_jobs)
                    for j in migrated_jobs:
                        migration[j[0]] += 1
                        if j[0] not in self.during_ovhd:
                            self.during_ovhd.add(j[0])
                        self.start_ovhd[j[0]] = self.Tnow
                    if len(migrated_jobs) > 0:
                        print(f'Promotion on GPU {gpu.index}, jobs {gpu.jobs}, slice {gpu.partition}', file=run_log, flush=True)
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in gpu_states]))
        
            ################ check if termination condition is met ################
        
            if sum(completion.values()) == len(completion) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {self.Tnow}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = self.Tnow 
                self.overall_rate.append(self.span_time)
                break
        
        ########################
        JCT = {}
        
        for job in self.job_runtime:
            if job not in self.filler_jobs:
                JCT[job] = self.comp_time[job] - self.queue_dict[job]

        return np.mean(list(JCT.values())), np.mean(self.overall_rate[:-1]), self.span_time

class oracle_paper(Oracle_Sim):
    def __init__(self, args):
        super().__init__(args)
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
            if j not in self.filler_jobs:
                completion[j] = 0
            migration[j] = 0
    
        run_log = open(f'/scratch/{user}/miso_logs/paper_eval/oracle{args.seed}.log','w')
        
        while True:
            # loop for getting jobs from the queue
            while queue_ind < len(queue) and self.queue_dict[queue[queue_ind]] <= self.Tnow:
                arrived_jobs.append(queue[queue_ind])
                queue_ind += 1
                
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
            for job in list(self.during_ovhd):
                if self.Tnow - self.start_ovhd[job] >= args.overhead:
                    self.during_ovhd.remove(job)
            # for each job currently running, reduce its remaining time proportionally
            emptied_gpu = []
            for gpu in gpu_states:
                slow_down = gpu.eval_degradation(self.perf_actual)
                for ind, job in enumerate(gpu.active_jobs):
                    if job in self.during_ovhd:
                        passed_time = 0
                    else:
                        passed_time = args.step / slow_down[ind]
                    if remain_time[job] - passed_time <= 0:
                        # job has finished, update GPU
                        if job in completion: completion[job] = 1 
                        real_ind = gpu.jobs.index(job)
                        gpu.jobs[real_ind] = 'idle'
                        self.comp_time[job] = self.Tnow
                        remain_time[job] = 0
                        emptied_gpu.append(gpu)
                        gpu.update_max_allowed(self.perf_actual)
                        print(f'Finish time: {self.Tnow}', file=run_log, flush=True)
                        print(f'job {job} finished', file=run_log, flush=True)
                    else:
                        if job in completion: completion[job] += passed_time / self.job_runtime[job]
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
                if 'idle' in gpu.jobs and len(arrived_jobs) == 0: 
                    num_mig, migrated_jobs = gpu.idle_partition_optimize_V2(self.perf_actual)
                    for j in migrated_jobs:
                        migration[j] += 1
                        if j not in self.during_ovhd:
                            self.during_ovhd.add(j)
                        self.start_ovhd[j] = self.Tnow

                    if len(gpu.active_jobs) > 0:
                        print(f'MIG re-partition on GPU {gpu.index}, jobs {gpu.jobs}, slice {gpu.partition}, {num_mig} migration counts', file=run_log, flush=True)
        
        #        if Tnow % 60 == 0
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in gpu_states]))
        
#            # sanity check
            for gpu in gpu_states:
                if 'idle' in gpu.jobs and gpu.max_allowed != '7g.40gb' and len(arrived_jobs) == 0:
                    raise RuntimeError('Check failed: GPU should not have bubble')
        
            ################ check if termination condition is met ################
        
            if sum(completion.values()) == len(completion) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {self.Tnow}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = self.Tnow 
                self.overall_rate.append(self.span_time)
                break
        
        ########################
        JCT = {}
        
        for job in self.job_runtime:
            if job not in self.filler_jobs:
                JCT[job] = self.comp_time[job] - self.queue_dict[job]

        return np.mean(list(JCT.values())), np.mean(self.overall_rate[:-1]), self.span_time

class miso_paper(Simulation):
    def __init__(self, args):
        super().__init__(args)

    def run(self, args):
        # GPU status class: records current partition, and jobs running on the partition
        gpu_states = []
        for i in range(args.num_gpu):
            gpu_states.append(GPU_status(i))
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
    
        run_log = open(f'/scratch/{user}/miso_logs/paper_eval/miso{args.seed}.log','w')
        
        while True:
            # loop for getting jobs from the queue
            while queue_ind < len(queue) and self.queue_dict[queue[queue_ind]] <= self.Tnow:
                arrived_jobs.append(queue[queue_ind])
                queue_ind += 1
                
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
            for job in list(self.during_ovhd):
                if self.Tnow - self.start_ovhd[job] >= args.overhead:
                    self.during_ovhd.remove(job)
            for gpu in list(self.during_mps):
                if self.Tnow - self.start_mps[gpu] >= args.mps_time:
                    self.during_mps.remove(gpu)
                    # start checkpointing for all jobs in this GPU
                    for job in gpu.active_jobs:
                        migration[job] += 1
                        if job in self.during_ovhd:
                            raise RuntimeError('Job cannot be in during ovhd when exiting MPS')
                        self.during_ovhd.add(job)
                        self.start_ovhd[job] = self.Tnow
                    print(f'MPS complete time: {self.Tnow}', file=run_log, flush=True)
                    print(f'jobs {str(gpu.active_jobs)} finished MPS on GPU {gpu.index}', file=run_log, flush=True)

            # for each job currently running, reduce its remaining time proportionally
            emptied_gpu = []
            for gpu in gpu_states:
                slow_down = gpu.eval_degradation(self.perf_actual)
                for ind, job in enumerate(gpu.active_jobs):
                    if job in self.during_ovhd:
                        passed_time = 0
                    elif gpu in self.during_mps:
                        passed_time = args.step / args.mps_rate
                    else:
                        passed_time = args.step / slow_down[ind]
                    if remain_time[job] - passed_time <= 0:
                        # job has finished, update GPU
                        if job in completion: completion[job] = 1 
                        real_ind = gpu.jobs.index(job)
                        gpu.jobs[real_ind] = 'idle'
                        self.comp_time[job] = self.Tnow
                        remain_time[job] = 0
                        emptied_gpu.append(gpu)
                        gpu.update_max_allowed(self.perf_pred)
                        print(f'Finish time: {self.Tnow}', file=run_log, flush=True)
                        print(f'job {job} finished', file=run_log, flush=True)
                    else:
                        if job in completion: completion[job] += passed_time / self.job_runtime[job]
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
                if 'idle' in gpu.jobs and len(arrived_jobs) == 0 and gpu not in self.during_mps:
                    Ts = time.time()
                    num_mig, migrated_jobs = gpu.idle_partition_optimize_V2(self.perf_pred, args.rand_gen)
                    Trun = time.time() - Ts
                    runtimes.append(Trun)
                    for j in migrated_jobs:
                        migration[j] += 1
                        if j not in self.during_ovhd:
                            self.during_ovhd.add(j)
                        self.start_ovhd[j] = self.Tnow

                    if len(gpu.active_jobs) > 0:
                        print(f'MIG re-partition on GPU {gpu.index}, jobs {gpu.jobs}, slice {gpu.partition}, {num_mig} migration counts', file=run_log, flush=True)
        
        #        if Tnow % 60 == 0
                cnt_active += len(gpu.active_jobs)
            active_jobs_per_gpu.append(cnt_active / args.num_gpu)
       
            self.overall_rate.append(sum([self.get_rate(gpu) for gpu in gpu_states]))
        
#            # sanity check
            for gpu in gpu_states:
                if 'idle' in gpu.jobs and gpu.max_allowed != '7g.40gb' and len(arrived_jobs) == 0 and gpu not in self.during_mps:
                    raise RuntimeError('Check failed: GPU should not have bubble')
        
            ################ check if termination condition is met ################
        
            if sum(completion.values()) == len(completion) and queue_ind == args.num_job and len(arrived_jobs) == 0:
                print(f'Time: {self.Tnow}, all jobs are finished!', file=run_log, flush=True)
                self.span_time = self.Tnow 
                self.overall_rate.append(self.span_time)
                break
        
        ########################
        JCT = {}
        
        for job in self.job_runtime:
            if job not in self.filler_jobs:
                JCT[job] = self.comp_time[job] - self.queue_dict[job]

        return np.mean(list(JCT.values())), np.mean(self.overall_rate[:-1]), self.span_time, np.max(runtimes)
