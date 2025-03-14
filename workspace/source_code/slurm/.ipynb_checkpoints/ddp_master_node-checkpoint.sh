MasterIPdress=$(hostname -i)
echo "Master IP Address: " $MasterIPdress
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
cd workspace
torchrun  --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=$MasterIPdress --master_port=1234 source_code/test_ddp.py