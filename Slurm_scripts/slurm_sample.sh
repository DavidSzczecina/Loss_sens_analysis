#!/bin/bash
#SBATCH --job-name=Loss_sens_analysis_Testing
#SBATCH --account=def-pfieguth
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err
#SBATCH --mail-user=dszczeci@uwaterloo.ca
#SBATCH --mail-type=ALL
  
# Activate your virtual environment and load required modules
source ../../envs/env3/bin/activate
module load python

# Change directory to your project folder (adjust the path as needed)
cd ~/projects/def-pfieguth/dszczeci/Loss_sens_analysis/python_files


DATASET="MNIST" # MNIST, FASHIONMNIST, CIFAR10, CIFAR100
CORRUPTION_RATE="0.2" # 0.05, 0.1, 0.15, 0.2


# Define a base command with parameters that remain constant
BASE_CMD="python3 main.py --supress_print --epochs 10 --num_folds 5 --dataset ${DATASET} --corruption_rate ${CORRUPTION_RATE}"



echo "Running baseline experiments for dataset: ${DATASET}, CR: ${CORRUPTION_RATE}"

echo "Running baseline experiments:"
# Run the base experiment (default loss cross entropy)
echo ">> Running cross entropy"
$BASE_CMD --loss ce
# Run the focal loss experiment
echo ">> Running focal loss"
$BASE_CMD --loss fl
# Run the GCELoss experiment
echo ">> Running GCELoss"
$BASE_CMD --loss gce
# Run the GCELoss experiment
echo ">> Running GCELoss"
$BASE_CMD --loss anl_ce
# Run the GCELoss experiment
echo ">> Running GCELoss"
$BASE_CMD --loss anl_fl



# Define arrays for the parameters you want to iterate over
delays=(0 2 4 6 8)
cutoffs=(0.005 0.01 0.025 0.05 0.10 0.20)

echo "Running experiments for Piecewise Zero (PZ) loss:"
# Loop over each delay and cutoff combination
for delay in "${delays[@]}"; do
  for cutoff in "${cutoffs[@]}"; do
    echo ">> Running with --delay ${delay} and --pz_cutoff ${cutoff}"
    $BASE_CMD --loss pz --delay ${delay} --pz_cutoff ${cutoff}
  done
done


# Define arrays for the parameters you want to iterate over
delays=(0 2 4 6 8)
bl_gammas=(0.1 0.2 0.3 0.4 0.5 0.6)

echo "Running experiments for Blurry loss:"
# Loop over each delay and cutoff combination
for delay in "${delays[@]}"; do
  for bl_gamma in "${bl_gammas[@]}"; do
    echo ">> Running with --delay ${delay} and --bl_gamma ${bl_gamma}"
    $BASE_CMD --loss blurry --delay ${delay} --bl_gamma ${bl_gamma}
  done
done

echo "All experiments submitted for dataset: ${DATASET}, CR: ${CORRUPTION_RATE}"