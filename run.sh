n_gpu=8
port=16006
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port $port ./main.py --world_size $n_gpu