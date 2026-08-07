echo "Loading common options"

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN



# ATTENTION_BACKEND defaults to 'auto' rather than 'flash': flash cannot serve a
# packed THD layout combined with context parallelism, and TE then raises
# "No dot product attention backend is available for the provided inputs" in the
# TRAINING forward -- i.e. after the whole rollout wave has already been
# collected, so each occurrence costs a full iteration. 'auto' lets TE choose a
# backend that can serve the inputs, and was validated numerically
# (inference/training logprob abs_diff 0.0059 vs 0.0053 for flash: within noise).
# Pin ATTENTION_BACKEND=flash for runs known to be unpacked and CP-free.
COMMON_OPTIONS="\
    --tensor-model-parallel-size $TP  \
    --pipeline-model-parallel-size $PP  \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --${PRECISION:-bf16} \
    --te-rng-tracker \
    --inference-dynamic-batching-buffer-size-gb ${INFERENCE_BUFFER_SIZE_GB:-20} \
    --data-parallel-random-init \
    --attention-backend ${ATTENTION_BACKEND:-auto} \
    --timing-log-level 1 \
    --log-timers-to-tensorboard \
    --save-retain-interval 160 \
    --inference-dynamic-batching-num-cuda-graphs 1 \
    --inference-dynamic-batching-unified-memory-level 1 \
    --adam-beta1 0.9 \
    --adam-beta2 ${ADAM_BETA2:-0.95} \
    --adam-eps 1e-8 \
    "

if [ ${LOWER_PRECISION:-false} == true ]; then
    echo "Lower precision experiments, disabling cuda graphs."
    ENABLE_CUDA_GRAPH=false
    COMMON_OPTIONS="${COMMON_OPTIONS} --no-gradient-accumulation-fusion"
else 
    COMMON_OPTIONS="${COMMON_OPTIONS}"
fi

if [ ${ENABLE_CUDA_GRAPH:-true} == true ]; then
    COMMON_OPTIONS="${COMMON_OPTIONS} --cuda-graph-impl=local --rl-persist-cuda-graphs"
fi
