#!/bin/bash

##################################################
##### Additional Instructions for Riselab Researchers
##################################################

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 2-00:00
#SBATCH --cpus-per-task 10
#SBATCH  --gres gpu:1
#SBATCH --nodelist=freddie

pwd
hostname
date
echo starting job...
# eval $(conda shell.bash hook)

source ~/.bashrc
cd /data/akristoffersen/anaconda
conda activate zffsr_akristoffersen_3_6
cd /home/eecs/akristoffersen/effsr/ZSSR

#  pip install -r ./Pseudo_Lidar_V2/requirements.txt
#  pip install pillow
#  pip install torch==1.0.0 torchvision==0.2.1
pip3 install tensorflow==1.5.1
pip3 install GPUtil
pip install opencv-python
pip install opencv-contrib-python

pip install scipy
pip install matplotlib

CUDA_VISIBLE_DEVICES=3

echo starting zssr:

export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

python3 run_ZSSR_video.py X2_REAL_CONF_VIDEO 3

echo ending zssr.
date