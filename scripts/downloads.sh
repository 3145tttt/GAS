#! /bin/bash

# EDM models
mkdir pretrained/edm

wget -O pretrained/edm/edm-cifar10-32x32-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
wget -O pretrained/edm/edm-ffhq-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
wget -O pretrained/edm/edm-afhqv2-64x64-uncond-vp.pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl


# LDM models
mkdir pretrained/ldm
mkdir pretrained/ldm/first_stage_models

wget -O pretrained/ldm/lsun_beds-256.zip https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip
unzip -d pretrained/ldm -o pretrained/ldm/lsun_beds-256.zip
mv pretrained/ldm/model.ckpt pretrained/ldm/lsun_beds-256.ckpt
rm pretrained/ldm/lsun_beds-256.zip

wget -O pretrained/ldm/cin256-v2.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt

wget -O pretrained/ldm/first_stage_models/vq-f4.zip https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
unzip -d pretrained/ldm/first_stage_models -o pretrained/ldm/first_stage_models/vq-f4.zip
mv pretrained/ldm/first_stage_models/model.ckpt pretrained/ldm/first_stage_models/vq-f4.ckpt
rm pretrained/ldm/first_stage_models/vq-f4.zip


# Stable Diffusion
mkdir pretrained/sd
wget -O pretrained/sd/v1-5-pruned-emaonly.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt


# FID reference statistics
mkdir fid-refs/edm
wget -O fid-refs/edm/afhqv2-64x64.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz
wget -O fid-refs/edm/ffhq-64x64.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz
wget -O fid-refs/edm/cifar10-32x32.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

mkdir fid-refs/ldm
wget -O fid-refs/ldm/VIRTUAL_lsun_bedroom256.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz
wget -O fid-refs/ldm/VIRTUAL_imagenet256_labeled.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

mkdir fid-refs/sd
wget --no-check-certificate -O fid-refs/sd/COCO_30k.npz https://drive.usercontent.google.com/download?id=1hlSlPbg1ycsPq7ZnYHBooVWuIiu3mAxe&confirm=t
