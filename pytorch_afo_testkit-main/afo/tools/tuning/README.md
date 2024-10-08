# EXPERIMENTAL pretune flow for pytorch

To collect the yaml required for this app, assuming `run.sh` run the workload.

`ROCBLAS_LAYER=4 bash run.sh 2>&1 | grep "\- { rocblas_function:" | uniq | tee rocblas.yaml`

To tune using 8 gpus run
`python3 tune_from_rocblasbench.py rocblas.yaml --cuda_device 0 1 2 3 4 5 6 7`

