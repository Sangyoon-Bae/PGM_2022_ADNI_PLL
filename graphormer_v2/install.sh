#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

## RUN ON NIPA SERVER ##

## modified by Stella Sangyoon Bae (stella.sangyoon.bae@gmail.com) ##

##########################################################################
#### first of all, you need to make conda environment with python 3.7 ####
##########################################################################


# install requirements
pip install torch==1.9.1+cu111 torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
conda install cudatoolkit=11.1 -c pytorch -c conda-forge
# install torchaudio, thus fairseq installation will not install newest torchaudio and torch(would replace torch-1.9.1)
pip install lmdb

pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-sparse==0.6.12 -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
pip install torch-geometric==1.7.2

pip install tensorboardX==2.4.1
#pip install ogb==1.3.2
pip install rdkit-pypi==2021.9.3
pip install dgl==0.7.2 -f https://data.dgl.ai/wheels/repo.html

cd fairseq
# if fairseq submodule has not been checkouted, run:
git submodule update --init --recursive
pip install . --use-feature=in-tree-build

# to solve fatal error: cub/cub.cuh: No such file or directory
git checkout dd3bd3c0497ae9a7ae7364404a6b0a4c501780b3

python setup.py build_ext --inplace
