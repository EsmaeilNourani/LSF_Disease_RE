#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH -p gpu
#SBATCH -t 72:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
### update with your project number and full path to where you have cloned the code
#SBATCH --account=
#SBATCH -o RegulaTome_extraction/TrainRelationExtractionSystem/cluster_logs/%j.out
#SBATCH -e RegulaTome_extraction/TrainRelationExtractionSystem/cluster_logs/%j.err

#1. folders and paths ...
DIR=$(pwd)
export RT_REL_FOLDERPATH=$(pwd)
model_address="$DIR/MODEL/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf/"
train_set_address="$DIR/RegulaTomeCorpus/train-set/"
devel_set_address="$DIR/RegulaTomeCorpus/devel-set/"
preds_model_output_address="$DIR/OUTPUTS/preds/$SLURM_JOBID"
logfile_address="$DIR/OUTPUTS/logs/${SLURM_JOBID}.log"

echo model_address=$model_address
echo devel_set_address=$devel_set_address
echo preds_model_output_address=$preds_model_output_address
echo logfile_address=$logfile_address

#create a prediction folder
mkdir -p $preds_model_output_address

# check number of input params
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 random_seed_index"
    exit 1
fi

random_seed_index="$1"

# set execution environment
module purge
module load pytorch/1.12
# update accordingly, give a folder on your machine
export TRANSFORMERS_CACHE="/transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run
python regulatome_final_rel_pipeline.py \
    --random_seed_index "$random_seed_index" \
    --model_address "$model_address" \
    --train_set_address "$train_set_address" \
    --devel_set_address "$devel_set_address" \
    --preds_model_output_address "$preds_model_output_address" \
    --logfile_address "$logfile_address"
