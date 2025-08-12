set -x

pkill -9 -f ray
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate mm-eureka
echo "conda activate mm-eureka"

# eval "$('/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv-hl/hadoop-basecv/mllm_env/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# echo "conda activate mm-eureka-hl"
# conda activate mm-eureka-hl
export http_proxy=http://10.70.23.3:8412 && export https_proxy=http://10.70.23.3:8412
# conda install scipy=1.15.3 -y
# conda install --force-reinstall numpy=1.26.4

# export http_proxy=http://10.253.34.172:6666
# export https_proxy=http://10.253.34.172:6666


# # CentOS
# sudo firewall-cmd --permanent --add-port=8265/tcp
# sudo firewall-cmd --reload

# eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mlm/yanfeng/software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# conda activate mm-r
model_tag=qwen25vl_7b_mm_retool_v8_3_codereward_light_sft
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/qwenvl25_outputs/qwen25vl_retool_full_sft_mm_code_1.8k_v3_zw_4x8_1e_bs64/full/sft/checkpoint-28/
DATA_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/data/Data_Mine/MM_ReTool/SFT_CoT_Data/0612/dataset_k12_filtered_retoolprompt3rl_for_qwen_instruct.jsonl
OUTPUT_DIR=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/mm_eureka_outputs/$model_tag
EXP_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA
cd $EXP_ROOT

export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8265
export NCCL_TIMEOUT=7200

export no_proxy="localhost,127.0.0.1"

export TOKENIZERS_PARALLELISM=false

# OUTPUT_DIR='/absolute/path/to/output/dir'

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD

JOB_ARGS=($(python get_job_args_lc.py))
echo "JOB_ARGS: ${JOB_ARGS[@]}"

NNODES="${JOB_ARGS[0]}"
GPUS_PER_NODE="${JOB_ARGS[1]}"
MASTER_ADDR="${JOB_ARGS[2]}"
MASTER_PORT="${JOB_ARGS[3]}"
NODE_RANK="${JOB_ARGS[4]}"

echo "NNODES: ${NNODES}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NODE_RANK: ${NODE_RANK}"


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


if [ "$NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head node..."
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    echo "Starting Ray worker node..."
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi
#examples/scripts/reward_func_qwen_instruct_retool_code_format.py
# examples/scripts/reward_func_qwen_instruct_retool_2.py
# /mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/anaconda3/envs/mm-eureka/bin/ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
  # --colocate_all_models \
if [ "$NODE_RANK" -eq 0 ]; then
  batch_size=128
  echo "Sleeping for 30 seconds..."
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir $WORKING_DIR \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --remote_rm_url examples/scripts/reward_func_qwen_instruct_retool_3.py \
    --actor_num_nodes 2 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 8 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_sleep \
    --vllm_gpu_memory_utilization 0.7 \
    --vllm_sync_backend nccl \
    --pretrain ${MODEL_PATH} \
    --save_path ${OUTPUT_DIR} \
    --micro_train_batch_size 2 \
    --train_batch_size $batch_size \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size $batch_size \
    --temperature 1.0 \
    --n_samples_per_prompt 8 \
    --lambd 1.0 \
    --gamma 1.0 \
    --max_epochs 1 \
    --num_episodes 10 \
    --prompt_max_len 3000 \
    --max_samples 100000 \
    --generate_max_len 10000 \
    --advantage_estimator group_norm \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.0 \
    --prompt_data ${DATA_PATH} \
    --disable_fast_tokenizer \
    --input_key message \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 20 \
    --ckpt_path "${OUTPUT_DIR}/ckpt" \
    --max_ckpt_num 1 \
    --enable_accuracy_filter \
    --accuracy_lower_bound 0.1 \
    --accuracy_upper_bound 0.9 \
    --save_hf_ckpt \
    --freeze_prefix visual \
    --use_tensorboard "${OUTPUT_DIR}/tensorboard" \
    --exe_code \
    --exe_code_actionmask \
    --load_checkpoint | tee ${OUTPUT_DIR}/training.log 
  # --exe_code
fi
ray stop

# --enable_accuracy_filter \
# --accuracy_lower_bound 0.1 \
# --accuracy_upper_bound 0.9 \