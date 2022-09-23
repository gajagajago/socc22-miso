# MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant GPU Clusters, 2022 ACM Symposium on Cloud Computing

## Experiment Setup
We run MISO on 4 GPU nodes, each node has 2 NVIDIA A100 GPUs. Sudo access is required.

On each GPU node, first copy the necessary files into memory.
`./copy_memory.sh`

On each GPU node, launch the gpu\_server.py script. This server listens on controller signals at port 10000.
`python gpu_server.py --node dxxx --host cxxx --tc miso`

On the scheduler node (CPU), run the experiment
`python run.py`

## MPS
manually enable mps by running `./enable_mps.sh`. Do this before starting `gpu_server.py`

Do it on raw GPU (with MIG disabled)

Then initialize MIG. To run in MPS mode, just set MIG to 7g.40gb

Not needed, but to disable MPS, just run `./disable_mps.sh` and manually find the PIDs and kill them

### gpu\_server.py
This script needs to be run on every GPU, it waits for signals sent from controller, and executes the command, 
for example reconfigure MIG, start/restart jobs

### mig\_helper.py
Import functions from this helper to configure MIG slices

### controller\_helper.py
Experiment helper, decides on action to do when received signal from jobs

## Note
This repo is still WIP.