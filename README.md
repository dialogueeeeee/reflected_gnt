export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 \
       --master_port=$(( RANDOM % 1000 + 50000 )) \
       train_scannet.py --config configs/gnt_scannet.txt \
       --ckpt_path ./out/gnt_best.pth --expname gnt_scannet_0811_sync_bn --val_set_list configs/scannetv2_test_split.txt --no_load_opt --no_load_scheduler