python3 eval.py --config configs/gnt_full.txt --eval_scenes orchids --expname gnt_full --chunk_size 500 --run_val --N_samples 192
python3 eval.py --config configs/gnt_llff.txt --eval_scenes orchids --expname gnt_llff --chunk_size 500 --run_val --N_samples 192

python3 eval.py --config configs/gnt_full.txt --eval_dataset rffr --eval_scenes art1 --expname gnt_full --chunk_size 500 --run_val --N_samples 192



python3 render.py --config configs/gnt_full.txt --eval_scenes orchids --expname gnt_full --chunk_size 500 --run_val --N_samples 192

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 ) \
       train.py --config configs/gnt_ft_rffr.txt --expname vanilla_Nray --N_rand 2048


export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$(( RANDOM % 1000 + 50000 ) train.py --config configs/gnt_ft_rffr.txt --expname low_Nray --N_rand 1024

