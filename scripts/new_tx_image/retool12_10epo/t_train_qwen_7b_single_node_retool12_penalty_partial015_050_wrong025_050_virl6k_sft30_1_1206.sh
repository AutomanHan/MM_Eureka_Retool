set -x

sh /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/MM_Eureka_Retool/scripts/mpi_init.sh
model_tag=new_qwenvl7b_retool12_251111/qwen25vl_sft30_singlenode_retool_12_penalty0_5_partial015_050_wrong025_050_virl6k_10epo_1_1207

MODEL_PATH=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/model_weight/Qwen2.5-VL-7B-Instruct
MODEL_PATH=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/model_weight/rubyla1/mm-retool-sft-mixed-s30
# DATA_PATH=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/RL_mmeureka_data_qhb/dataset_k12_filtered_retoolprompt3_for_qwen_instruct_tx.jsonl
DATA_PATH=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/data/MM_retool_sft_rl_data/RL_data_ruby/base_vl_38k_pr_required_codes_rltrain.jsonl
OUTPUT_DIR=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/mm_retool_output/$model_tag
EXP_ROOT=/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/MM_Eureka_Retool
cd $EXP_ROOT

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
export WORKING_DIR=$PWD

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


sleep 30
ray status
if [ $MASTER_IP = $LOCAL_IP]; then
  # readonly MASTER_ADDR="127.0.0.1"
  MASTER_ADDR="$__POD_IP__"
else
  readonly MASTER_ADDR="$__POD_IP__" # gemini-2
fi
# MASTER_ADDR=0.0.0.0
echo "Master address: $MASTER_ADDR"
MASTER_PORT=7689
TOKENIZERS_PARALLELISM=true 
# RAY_ADDRESS="http://$MASTER_ADDR:$RAY_DASHBOARD_PORT" 
ray job submit \
--working-dir $WORKING_DIR \
--address "$MASTER_ADDR:$MASTER_PORT" \
-- python3 -m openrlhf.cli.train_ppo_ray \
--ref_num_nodes 1 \
--ref_num_gpus_per_node 8 \
--remote_rm_url examples/scripts/reward_func_qwen_instruct_retool_12.py \
--actor_num_nodes 1 \
--actor_num_gpus_per_node 8 \
--vllm_num_engines 8 \
--vllm_tensor_parallel_size 1 \
--colocate_all_models \
--vllm_enable_sleep \
--vllm_gpu_memory_utilization 0.3 \
--vllm_sync_backend nccl \
--pretrain ${MODEL_PATH} \
--save_path ${OUTPUT_DIR} \
--micro_train_batch_size 4 \
--train_batch_size 64 \
--micro_rollout_batch_size 4 \
--rollout_batch_size 64 \
--temperature 1.0 \
--n_samples_per_prompt 8 \
--lambd 1.0 \
--gamma 1.0 \
--max_epochs 1 \
--num_episodes 10 \
--prompt_max_len 3000 \
--max_samples 100000 \
--generate_max_len 4096 \
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
--save_steps 50 \
--ckpt_path "${OUTPUT_DIR}/ckpt" \
--max_ckpt_num 1 \
--enable_accuracy_filter \
--accuracy_lower_bound 0.1 \
--accuracy_upper_bound 0.9 \
--save_hf_ckpt \
--freeze_prefix visual \
--use_tensorboard "${OUTPUT_DIR}/tensorboard" \
--exe_code \
--code_penalty_group \
--code_penalty_patial 0.15 \
--code_penalty_patial_high 0.50 \
--code_penalty_wrong_patial_low 0.25 \
--code_penalty_wrong_patial_high 0.50 \
--load_checkpoint | tee ${OUTPUT_DIR}/training.log
#fi
ray stop
# --exe_code_actionmask \
# --enable_accuracy_filter \
# --accuracy_lower_bound 0.1 \
# --accuracy_upper_bound 0.9 \
