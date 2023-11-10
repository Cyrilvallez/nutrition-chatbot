#!/bin/bash

#SBATCH --job-name=nutrition-chatbot
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:2
#SBATCH --chdir=/cluster/raid/home/vacy/nutrition-chatbot

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate llm

# Note: the -u option is absolutely necesary here to force the flush of the link 
# to connect to the app!
python3 -u chatbot.py

conda deactivate
