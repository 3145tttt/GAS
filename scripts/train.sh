#! /bin/bash

config_name="configs/edm/cifar10.yaml"
loss_type="GS"
NFE=4
train_size=1400

python main.py \
    --config=$config_name \
    --loss_type=$loss_type \
    --student_step=$NFE \
    --train_size=$train_size
