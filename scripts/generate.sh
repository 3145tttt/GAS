#! /bin/bash

NFE=4
checkpoint='YOUR CHECKPOINT'
out_dir="CIFAR10_NFE_4"
ITER_NUM=10000


torchrun --standalone --nproc_per_node=1 generate.py \
	--config="configs/edm/cifar10.yaml" \
	--outdir=${out_dir} \
	--seeds=50000-99999 \
	--batch=1024 \
	--steps=${NFE} \
	--checkpoint_path=checkpoints/${checkpoint}/${ITER_NUM}.pt

torchrun --standalone --nproc_per_node=1 fid.py calc \
	--images=${out_dir} \
	--ref=fid-refs/edm/cifar10-32x32.npz \
	--batch=1024 \
	--num=50000 >> FID_${out_dir}.txt

