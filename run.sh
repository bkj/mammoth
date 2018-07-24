#!/bin/bash

# run.sh

mkdir results
python experiments/mnist-mlp.py | tee results/mnist-mlp.py

# >>

python experiments/test.py

# <<