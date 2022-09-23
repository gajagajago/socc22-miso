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

class Static_Sim(Simulation):

    def __init__(self, args):
        super().__init__(args)

    def try_schedule(self, job, gpu_list, migration, run_log):
        sched_done = False

        for instance in GPU_status.n2s_reverse.values():
            if instance in self.perf_pred[job%100]:
                min_size = instance
                break
        allowed_slices = []
        for code, instance in GPU_status.num_to_str.items():
            if code >= int(min_size.split('g.')[0]):
                allowed_slices.append(instance)

        gpu_avail = [g for g in gpu_list if g.max_allowed != 'full']

        # sort by maximum idle slice size
        sorted_gpus = sorted(gpu_avail, key=lambda x: x.max_inactive_slice[0], reverse=True)

        if len(sorted_gpus) > 0:
            gpu = sorted_gpus[0]
            slice_code, slice_ind = gpu.max_inactive_slice
            if GPU_status.num_to_str[slice_code] in allowed_slices:
                gpu.jobs[slice_ind] = job
                gpu.static_max_slice()
                sched_done = True
                print(f'Schedule time: {self.Tnow}', file=run_log, flush=True)
                print(f'job {job} scheduled on GPU {gpu.index}, partition {gpu.partition}, jobs {gpu.jobs}', file=run_log, flush=True)

        return sched_done

    def run(self, args, slice_code=1, partition=[4,2,1]):
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
    
        run_log = open('logs/simulation_static.log','w')
        
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
        Path('logs/static').mkdir(parents=True, exist_ok=True)
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
            with open(f'logs/static/{name}.json', 'w') as f:
                json.dump(metric, f, indent=4)
        migration['average'] = np.mean(list(migration.values()))
        
        with open('logs/static/active_jobs_per_gpu.json', 'w') as f:
            json.dump(active_jobs_per_gpu, f, indent=4)
        with open('logs/static/remain_time.json', 'w') as f:
            json.dump(remain_time, f, indent=4)
        with open('logs/static/completion.json', 'w') as f:
            json.dump(completion, f, indent=4)
        with open('logs/static/progress.json', 'w') as f:
            json.dump(progress, f, indent=4)
        with open('logs/static/migration.json', 'w') as f:
            json.dump(migration, f, indent=4)
        with open('logs/static/overall_rate.json', 'w') as f:
            json.dump(self.overall_rate, f, indent=4)


