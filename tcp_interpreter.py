import pdb
import time

def interpret_full(data_str, runtime, run_log):
    if 'finish' in data_str: # job 10 finish
        jobid = int(data_str.split(' ')[1])
        runtime.finish[jobid] = 1
        runtime.completion[jobid] = 1
        runtime.comp_time[jobid] = int(time.time())
        # find the GPU and slice
        gpuid, sliceid = runtime.job_exe[jobid]
        runtime.gpu_states[gpuid].jobs[sliceid] = 'idle'
        runtime.emptied_gpu[runtime.gpu_states[gpuid]] = int(time.time())
        runtime.gpu_states[gpuid].max_allowed = '7g.40gb'
        print(f'Finish time: {int(time.time()-runtime.start_time)}', file=run_log, flush=True)
        print(f'job {jobid} finished on GPU {gpuid}', file=run_log, flush=True)
    elif 'completion' in data_str: # job 10 completion 0.25
        jobid = int(data_str.split(' ')[1])
        compl = float(data_str.split(' ')[3])
        runtime.completion[jobid] = compl
    elif 'pid' in data_str: # job 10 pid 3333
        jobid = int(data_str.split(' ')[1])
        pid = data_str.split(' ')[3]
        runtime.pid_dict[jobid] = pid

def interpret_mps(data_str, runtime, run_log):
    if 'finish' in data_str: # job 10 finish
        jobid = int(data_str.split(' ')[1])
        runtime.finish[jobid] = 1
        runtime.completion[jobid] = 1
        runtime.comp_time[jobid] = int(time.time())
        # find the GPU and slice
        gpuid, sliceid = runtime.job_exe[jobid]
        runtime.gpu_states[gpuid].jobs.remove(jobid)
        runtime.emptied_gpu[runtime.gpu_states[gpuid]] = int(time.time())
        print(f'Finish time: {int(time.time()-runtime.start_time)}', file=run_log, flush=True)
        print(f'job {jobid} finished on GPU {gpuid}', file=run_log, flush=True)
    elif 'completion' in data_str: # job 10 completion 0.25
        jobid = int(data_str.split(' ')[1])
        compl = float(data_str.split(' ')[3])
        runtime.completion[jobid] = compl
    elif 'pid' in data_str: # job 10 pid 3333
        jobid = int(data_str.split(' ')[1])
        pid = data_str.split(' ')[3]
        runtime.pid_dict[jobid] = pid

def interpret_static(data_str, runtime, run_log):
    if 'finish' in data_str: # job 10 finish
        jobid = int(data_str.split(' ')[1])
        runtime.finish[jobid] = 1
        runtime.completion[jobid] = 1
        runtime.comp_time[jobid] = int(time.time())
        # find the GPU and slice
        gpuid, sliceid = runtime.job_exe[jobid]
        if runtime.gpu_states[gpuid].jobs[sliceid] != jobid:
            raise RuntimeError('Error: mismatch between GPU state and job_exe dict')
        runtime.gpu_states[gpuid].jobs[sliceid] = 'idle'
        runtime.emptied_gpu[runtime.gpu_states[gpuid]] = int(time.time())
        runtime.gpu_states[gpuid].static_max_slice()
        print(f'Finish time: {int(time.time()-runtime.start_time)}', file=run_log, flush=True)
        print(f'job {jobid} finished on GPU {gpuid}', file=run_log, flush=True)
    elif 'completion' in data_str: # job 10 completion 0.25
        jobid = int(data_str.split(' ')[1])
        compl = float(data_str.split(' ')[3])
        runtime.completion[jobid] = compl
    elif 'pid' in data_str: # job 10 pid 3333
        jobid = int(data_str.split(' ')[1])
        pid = data_str.split(' ')[3]
        runtime.pid_dict[jobid] = pid
    elif 'ckpt' in data_str: # ckpt job 10 batch 1000
        jobid = int(data_str.split(' ')[2])
        resume_batch = int(data_str.split(' ')[4])
        runtime.ckpt_dict[jobid] = 1
        runtime.ckpt_start[jobid] = int(time.time())
        runtime.ckpt_batch[jobid] = resume_batch
        gpuid, sliceid = runtime.job_exe[jobid]
        runtime.ckpt_buffer[runtime.gpu_states[gpuid]] = int(time.time())
        print(f'Checkpoint time: {int(time.time()-runtime.start_time)}', file=run_log, flush=True)
        print(f'job {jobid} checkpointed on GPU {gpuid}', file=run_log, flush=True)
    elif 'recover' in data_str: # recover job 10
        jobid = int(data_str.split(' ')[2])
        if runtime.ckpt_dict[jobid] == 1:
            recover_time = int(time.time())
            runtime.ckpt_ovhd[jobid].append(recover_time - runtime.ckpt_start[jobid])
            runtime.ckpt_dict[jobid] = 0

def interpret_miso(data_str, runtime, run_log):
    if 'finish' in data_str: # job 10 finish
        jobid = int(data_str.split(' ')[1])
        runtime.finish[jobid] = 1
        runtime.completion[jobid] = 1
        runtime.comp_time[jobid] = int(time.time())
        # find the GPU and slice
        gpuid, sliceid = runtime.job_exe[jobid]
        runtime.gpu_states[gpuid].jobs[sliceid] = 'idle'
        runtime.emptied_gpu[runtime.gpu_states[gpuid]] = int(time.time())
        runtime.gpu_states[gpuid].update_max_allowed(runtime.perf_actual) 
        print(f'Finish time: {int(time.time()-runtime.start_time)}', file=run_log, flush=True)
        print(f'job {jobid} finished on GPU {gpuid}', file=run_log, flush=True)
    elif 'completion' in data_str: # job 10 completion 0.25
        jobid = int(data_str.split(' ')[1])
        compl = float(data_str.split(' ')[3])
        runtime.completion[jobid] = compl
    elif 'pid' in data_str: # job 10 pid 3333
        jobid = int(data_str.split(' ')[1])
        pid = data_str.split(' ')[3]
        runtime.pid_dict[jobid] = pid
    elif 'ckpt' in data_str: # ckpt job 10 batch 1000
        jobid = int(data_str.split(' ')[2])
        resume_batch = int(data_str.split(' ')[4])
        runtime.ckpt_dict[jobid] = 1
        runtime.ckpt_start[jobid] = int(time.time())
        runtime.ckpt_batch[jobid] = resume_batch
        gpuid, sliceid = runtime.job_exe[jobid]
        runtime.ckpt_buffer[runtime.gpu_states[gpuid]] = int(time.time())
        print(f'Checkpoint time: {int(time.time()-runtime.start_time)}', file=run_log, flush=True)
        print(f'job {jobid} checkpointed on GPU {gpuid}', file=run_log, flush=True)
    elif 'recover' in data_str: # recover job 10
        jobid = int(data_str.split(' ')[2])
        if runtime.ckpt_dict[jobid] == 1:
            recover_time = int(time.time())
            runtime.ckpt_ovhd[jobid].append(recover_time - runtime.ckpt_start[jobid])
            runtime.ckpt_dict[jobid] = 0

