#!/bin/bash
eval "$('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/lanxiaohan/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate mm-eureka
echo "conda activate mm-eureka"

EXP_ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA
cd $EXP_ROOT

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
# 从命令行参数获取NODE_RANK，如果没有则从环境变量获取
if [ $# -gt 0 ]; then
    NODE_RANK=$1
else
    NODE_RANK=${NODE_RANK:-0}
fi

echo "当前NODE_RANK值: $NODE_RANK"

case $NODE_RANK in
    0)
        echo "执行NODE_RANK=0的任务..."
        bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0717/qwen25_retool_run_0724_sft_w_mixed_data_4k_rl_w_k12_8k_bs64_reward_v7_early_bonus_ckpt_30.sh
        # python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/qiuhaibo/workspace/tools/scripts/gpu_util.py
        # bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0702/retool_run_0702_bs128_reward_v4.sh
        ;;
    1)
        echo "执行NODE_RANK=1的任务..."
        bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0717/qwen25_retool_run_0724_base_rl_k12_8k_bs64_reward_v7_early_bonus.sh
        # python3 /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/qiuhaibo/workspace/tools/scripts/gpu_util.py
        # bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0702/retool_run_0702_bs64_reward_v5_mixed_sft_k12_rl.sh
        # bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0702/retool_run_0702_bs128_reward_v4.sh
        ;;
    2)
        echo "执行NODE_RANK=2的任务..."
        bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0717/qwen25_retool_run_0717_sft_w_mixed_data_4k_rl_w_k12_8k_bs64_reward_v4_ckpt_30.sh
        # bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0702/retool_run_0702_bs64_reward_v4.sh
        ;;
    3)
        echo "执行NODE_RANK=3的任务..."
        bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0717/qwen25_retool_run_0723_sft_w_mixed_data_4k_rl_w_k12_8k_bs64_reward_v6_ckpt_30.sh
        # bash /mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lanxiaohan/reasoning_hc/code/MM-EUREKA/scripts/mm_retool_lxh/0702/retool_run_0702_bs64_reward_v5.sh
        ;;
    *)
        echo "错误: 不支持的NODE_RANK值: $NODE_RANK"
        echo "用法: $0 [1|2|3|4] 或设置环境变量 NODE_RANK"
        exit 1
        ;;
esac

echo "NODE_RANK=$NODE_RANK 的任务执行完成"
