
## Data preparing 
mkdir data
python scripts/download_scannetv2.py --out_dir data # wait until all data were downloaded

bash scripts/reader.sh


## For JuJu running

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_scannet_0811_sync_bn --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler