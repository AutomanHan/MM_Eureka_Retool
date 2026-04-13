#!/usr/bin/env bash
#

set -ex

echo "============================environment========================="
env | while IFS='' read -r line; do
  echo "$line"
done
echo "================================================================"

# for local test
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
if [[ -z "${__POD_IP__}" ]]; then
  readonly MASTER_ADDR="127.0.0.1"
else
  readonly MASTER_ADDR="$__POD_IP__" # gemini-2
fi
echo "MASTER_ADDR: $MASTER_ADDR"
readonly MASTER_PORT="7689"
readonly RANK=0

cp /etc/mpi/hostfile .
# 使用sed命令修改文件内容
sed -i 's/=8/=1/g' hostfile
HOSTFILE=`pwd`/hostfile
if [ -f $HOSTFILE ]; then
  readonly NNODES=`wc -l $HOSTFILE | awk '{print $1}'` # gemini-2
else
  readonly NNODES=1
fi

mpirun -v --allow-run-as-root --bind-to none --map-by slot -mca routed direct --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1 echo "mpirun-test--"

GEMINI_MPI_ARGS='--hostfile '$HOSTFILE' --bind-to none --map-by slot --mca routed direct --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH'

echo $NNODES $MASTER_ADDR

mpirun \
  -v --allow-run-as-root \
  $GEMINI_MPI_ARGS \
  sh /mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_4/nathanchan/projects/retool/code/MM_Eureka_Retool/scripts/ray_init.sh \
  $MASTER_ADDR \
  $MASTER_PORT

sleep 30
ray status