mkdir -p $(dirname $2)

NGPUS=$(nvidia-smi -L | wc -l)
mpirun  -allow-run-as-root -np ${NGPUS} -H localhost:${NGPUS} -x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=23457  -x HOROVOD_TIMELINE -x OMP_NUM_THREADS=1 -x KMP_AFFINITY='granularity=fine,compact,1,0'  -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4  --report-bindings $1  2>&1 | tee -a $2
