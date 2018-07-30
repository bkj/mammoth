#!/bin/bash

# run.sh

# --
# Download data

mkdir -p data
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz && rm aclImdb_v1.tar.gz
mv aclImdb ./data/aclImdb

# --
# Run

python prep.py

mkdir results

# --
# Tests

# Same result every time
python nbsgd.py --verbose --untrain --fixed-data --num-iters 10

# Variability due to minibatches
python nbsgd.py --verbose --untrain --num-iters 10

# --
# Experiments (round 1)

# Regular training (w/ mildly tuned parameters)
CUDA_VISIBLE_DEVICES=0 python nbsgd.py --verbose \
    --mo-init -2 | tee results/normal-2.jl
# {"meta_iter": 19, "train_acc": 0.996875, "val_acc": 0.9218400120735168, "test_acc": 0.9215199947357178, "time": 63.3641152381897}

# Learn everything
CUDA_VISIBLE_DEVICES=1 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-lrs \
    --learn-mos \
    --learn-meta \
    --mo-init -2 \
    --hyper-lr 0.1 | tee results/all.jl
# {"meta_iter": 19, "train_acc": 0.969140625, "val_acc": 0.9961599707603455, "test_acc": 0.9254400134086609, "time": 379.6041805744171}

# Learn just meta+init
CUDA_VISIBLE_DEVICES=2 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-meta \
    --mo-init -2 \
    --hyper-lr 0.1 | tee results/meta.jl
# {"meta_iter": 19, "train_acc": 0.964453125, "val_acc": 0.9964799880981445, "test_acc": 0.924560010433197, "time": 387.658460855484}

# Just lr+mo
CUDA_VISIBLE_DEVICES=3 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-lrs \
    --learn-mos \
    --mo-init -2 \
    --hyper-lr 0.1 | tee results/lrmo.jl
# {"meta_iter": 19, "train_acc": 0.971875, "val_acc": 0.9913600087165833, "test_acc": 0.9250400066375732, "time": 225.99636697769165}

# --
# Experiments (round 2)

# Regular training (w/ mildly tuned parameters)
CUDA_VISIBLE_DEVICES=0 python nbsgd.py --verbose \
    --mo-init -1 \
    --lr-init 0.1 | tee results/normal-lr10-mo90.jl

# Learn everything
CUDA_VISIBLE_DEVICES=1 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-lrs \
    --learn-mos \
    --learn-meta \
    --lr-init 0.1 \
    --mo-init -1 \
    --hyper-lr 0.1 | tee results/all-lr10-mo90.jl

# Learn just meta+init
CUDA_VISIBLE_DEVICES=2 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-meta \
    --lr-init 0.1 \
    --mo-init -1 \
    --hyper-lr 0.1 | tee results/meta-lr10-mo90.jl

# Just lr+mo
CUDA_VISIBLE_DEVICES=3 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-lrs \
    --learn-mos \
    --lr-init 0.1 \
    --mo-init -1 \
    --hyper-lr 0.1 | tee results/lrmo-lr10-mo90.jl


# --
# Experiments (round 3)

CUDA_VISIBLE_DEVICES=0 python nbsgd.py --verbose \
    --one-r \
    --mo-init -2 | tee results/normal-oner.1.jl



# Learn everything
CUDA_VISIBLE_DEVICES=1 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-lrs \
    --learn-mos \
    --learn-meta \
    --mo-init -2 \
    --one-r \
    --hyper-lr 0.1 | tee results/all-oner.jl

# Learn just meta+init
CUDA_VISIBLE_DEVICES=2 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-meta \
    --mo-init -2 \
    --one-r \
    --hyper-lr 0.1 | tee results/meta-oner.jl

# Just lr+mo
CUDA_VISIBLE_DEVICES=3 python nbsgd.py --verbose \
    --untrain \
    --learn-init \
    --learn-lrs \
    --learn-mos \
    --mo-init -2 \
    --one-r \
    --hyper-lr 0.1 | tee results/lrmo-oner.1.jl





