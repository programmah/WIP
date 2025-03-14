#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --partition=gpu         # GPU partition
#SBATCH --ntasks-per-node=4     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --threads-per-core=1    # number of threads per core
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=12:00:00         # format: HH:MM:SS
##SBATCH --nodelist=dgx06,dgx08
#SBATCH --exclusive
#SBATCH --mem 0
#SBATCH --job-name=ddpslurm
#SBATCH -o %x.output%j
#SBATCH -e %x.error%j
##SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=none
#SBATCH --overcommit
#SBATCH --parsable


# load conda env

#module load conda/2023
#conda init
#source activate env_mnode

# get master node IP Address 

SERVER="`hostname`"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST"))
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=12340
echo "Master IP address: " $MASTER_ADDR

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
cd workspace
srun python source_code/slurm_ddp.py
#srun torchrun --nproc_per_node=4 --nnodes=2 --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:29500 source_code/slurm_ddp.py