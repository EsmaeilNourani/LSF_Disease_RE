#!/bin/sh
### Account information
#PBS -W group_list=
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -o ${PBS_JOBID}.out
#PBS -e ${PBS_JOBID}.err
### Number of nodes
#PBS -l nodes=1:ppn=24:gpus=1
### Memory
#PBS -l mem=32gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=23:00:00



#activate env
conda activate RE 


echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

#1. folders and paths ...
DIR=$PBS_O_WORKDIR
export RT_REL_FOLDERPATH=$PBS_O_WORKDIR


model_address="$DIR/MODEL/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf/"
train_set_address="$DIR/LSD600Corpus/train-set/"
devel_set_address="$DIR/LSD600Corpus/devel-set/"
preds_model_output_address="$DIR/OUTPUTS/preds/${PBS_JOBID}"
logfile_address="$DIR/OUTPUTS/logs/${PBS_JOBID}.log"

echo model_address=$model_address
echo devel_set_address=$devel_set_address
echo preds_model_output_address=$preds_model_output_address
echo logfile_address=$logfile_address

#create a prediction folder
mkdir -p $preds_model_output_address


random_seed_index=1  # Can be 0 to 3

# update accordingly, give a folder on your machine
export TRANSFORMERS_CACHE="transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run
python LSF_DIS_rel_pipeline.py \
    --random_seed_index "$random_seed_index" \
    --model_address "$model_address" \
    --train_set_address "$train_set_address" \
    --devel_set_address "$devel_set_address" \
    --preds_model_output_address "$preds_model_output_address" \
    --logfile_address "$logfile_address"
