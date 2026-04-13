# !/bin/bash

set -x

MASTER_IP=$1
MASTER_PORT=$2
LOCAL_IP=${__HOST_IP__}
NODE_RANK=$OMPI_COMM_WORLD_RANK


LOG_PATH=ray_log
rm -rf $LOG_PATH
mkdir $LOG_PATH
if [ $MASTER_IP = $LOCAL_IP ]; then
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8080 --object-store-memory=798863917056 --node-ip-address=$MASTER_IP --port=$MASTER_PORT --block > ${LOG_PATH}/${POD_NAME}.${__HOST_IP__}.log 2>&1 &
else
    ray start --address ${MASTER_IP}:${MASTER_PORT} --runtime-env-agent-port 28629 --block > ${LOG_PATH}/${POD_NAME}_${__HOST_IP__}.log 2>&1 &
fi
