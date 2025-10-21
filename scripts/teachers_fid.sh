#! /bin/bash

# EDM
torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=data/teachers/edm/cifar10/images \
    --ref=fid-refs/edm/cifar10-32x32.npz \
    --batch=1024

torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=data/teachers/edm/ffhq/images \
    --ref=fid-refs/edm/ffhq-64x64.npz \
    --batch=1024

torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=data/teachers/edm/afhqv2/images \
    --ref=fid-refs/edm/afhqv2-64x64.npz \
    --batch=1024


# LDM
torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=data/teachers/ldm/lsun_beds256/images \
    --ref=fid-refs/ldm/VIRTUAL_lsun_bedroom256.npz \
    --batch=1024 

torchrun --standalone --nproc_per_node=1 fid.py calc \
    --images=data/teachers/ldm/cin256-v2/images \
    --ref=fid-refs/ldm/VIRTUAL_imagenet256_labeled.npz \
    --batch=1024 


NFEs=(5 6 7 8)
for NFE in "${NFEs[@]}"; do
    torchrun --standalone --nproc_per_node=1 fid.py calc \
        --images=data/teachers/sd-v1/nfe=${NFE}/images \
        --ref=fid-refs/sd/COCO_30k.npz \
        --num=30000 \
        --batch=1024 
done
