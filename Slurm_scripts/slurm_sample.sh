#!/bin/bash
#SBATCH --job-name=bioscan_cleaning
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

# Define a base command with parameters that remain constant
BASE_CMD="python3 loss_sens_analysis.py --basicModel --supress_print --epochs 10 --num_folds 5 --corruption_rate 0.2 --blurry_loss_gamma 0.0"

echo "Running base experiments:"

# Run the base experiment (default loss, e.g., cross entropy)
echo ">> Running default loss"
$BASE_CMD

# Run the focal loss experiment
echo ">> Running focal loss"
$BASE_CMD --focalLoss

# Run the GCELoss experiment
echo ">> Running GCELoss"
$BASE_CMD --GCELoss

# Now define arrays for the parameters you want to iterate over
delays=(0 2 4 6 8)
cutoffs=(0.005 0.01 0.025 0.05 0.10 0.20)

echo "Running experiments for various delays and cutoff_pt values:"
# Loop over each delay and cutoff combination
for delay in "${delays[@]}"; do
  for cutoff in "${cutoffs[@]}"; do
    echo ">> Running with --delay ${delay} and --cutoff_pt ${cutoff}"
    $BASE_CMD --delay ${delay} --cutoff_pt ${cutoff}
  done
done

echo "All experiments submitted."