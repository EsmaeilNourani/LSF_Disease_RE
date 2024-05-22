#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
### update with your project number and full path to where you have cloned the code
#SBATCH --account=
#SBATCH -o RegulaTome_extraction/LargeScaleRelationExtractionPipeline/cluster_logs/%j.out
#SBATCH -e RegulaTome_extraction/LargeScaleRelationExtractionPipeline/cluster_logs/%j.err

#get current working directory
DIR=$(pwd)

# delete job file on exit
function on_exit {
    rm -f jobs/$SLURM_JOBID
}
trap on_exit EXIT

# check number of input params
if [ "$#" -ne 1 ]; then
    echo "incorrect params"
    exit 1
fi

folder_index="$1"

# create/recreate output directory
rm -rf "outputs/output_${folder_index}"
mkdir "outputs/output_${folder_index}"

# set execution environment
module purge
module load pytorch
# UPDATE: update accordingly, give a folder on your machine
export TRANSFORMERS_CACHE="transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run the system on the cluster. Receives one folder as input, processess all .tar.gz files inside that folder ...
# UPDATE: MAKE SURE --input_folder_path actually points to a folder like input_1, input_2, ...
python run_ls_pipeline.py \
    --configs_file_path "${DIR}/data_regulatome_final_configs.json" \
    --log_file_path "${DIR}/logs/${folder_index}.log" \
    --model_folder_path "${DIR}/model" \
    --input_folder_path "${DIR}/inputs/input_${folder_index}" \
    --output_folder_path "${DIR}/outputs/output_${folder_index}"

