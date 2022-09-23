### collect\_mps.py:
This script is responsible for collecting MPS performance as training data

It starts a gRPC server with a local port, when a job script is ready (1 batch is trained), the script will try to connect to this port and notify the server. 
When the server has received ready signal from all launched jobs on MPS, it sends an interrupt to each job process so the job starts execution.

The job runs for a limited time specified by input argument. When the time limit is reached, it interrupts itself and saves the batch time recorded in the callback function.

### run\_enumerate.py:
This script is to collect MIG performance as label data. The job starts on a particular MIG slice. The gRPC is used to automatically send signal to start measurement once the job is ready. The gRPC server is configured differently since there is no need for synchronization in MIG (interference free.)
