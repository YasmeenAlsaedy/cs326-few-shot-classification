#!/bin/bash
#SBATCH --job-name=protonet
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --time=20:00:00

module load gcc/6.4.0

cd /ibex/scratch/alsaedyy/cs326_projects/cs326-few-shot-classification


CUDA_VISIBLE_DEVICES=0 python run_experiment.py -m protonet

