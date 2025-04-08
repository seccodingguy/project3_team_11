#!/bin/bash

sudo apt update

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

cat <<EOT >> ~/.bashrc
export PATH="$PATH:~/miniconda3/bin"
EOT

source ~/.bashrc

conda init

conda config --add channels conda-forge

conda create --name ai_dev python=3.10

conda activate ai_dev

conda install cudnn cudatoolkit numpy scipy tensorflow transformers keras pytorch scikit-learn pandas xgboost nltk spacy opencv plotly seaborn

conda install torch torchvision

conda install -c nvidia cuda-nvcc==11.*