
## Data preparing 
mkdir data
python scripts/download_scannetv2.py --out_dir data # wait until all data were downloaded

bash scripts/reader.sh


## For training

export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet_selected_inds.txt \
       --ckpt_path ./out/gnt_best.pth --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler


export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet_selected_inds_scratch.txt \
       --val_set_list configs/scannetv2_test_split.txt


export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet_selected_inds_auxloss.txt \
       --ckpt_path ./out/gnt_best.pth --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler