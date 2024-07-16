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

# create/recreate output directory
mkdir "outputs/sample_data"
mkdir "outputs/sample_data/LSD600Corpus"

# update accordingly, give a folder on your machine
export TRANSFORMERS_CACHE="/transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run the system on the cluster. Receives one folder as input, processess all .tar.gz files inside that folder ...
# if you want to run the system on your own files, you can follow below instructions , once you put your files ...


python run_ls_pipeline.py \
    --configs_file_path "${DIR}/LSF_DIS_rel_configs.json" \
    --log_file_path "${DIR}/logs/test_data.log" \
    --model_folder_path "${DIR}/model" \
    --input_folder_path "${DIR}/sample_data/LSD600Corpus" \
    --output_folder_path "${DIR}/outputs/sample_data/LSD600Corpus" \
