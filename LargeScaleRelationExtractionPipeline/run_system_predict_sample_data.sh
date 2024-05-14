#!/bin/bash
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

# create/recreate output directory
rm -rf "outputs/sample_data"
mkdir "outputs/sample_data"
mkdir "outputs/sample_data/input_1"
mkdir "outputs/sample_data/input_2"

# set execution environment
module purge
module load pytorch
# update accordingly, give a folder on your machine
export TRANSFORMERS_CACHE="/transformers_cache/"
export TOKENIZERS_PARALLELISM="false"

# run the system on the cluster. Receives one folder as input, processess all .tar.gz files inside that folder ...
# if you want to run the system on your own files, you can follow below instructions , once you put your files ...

python run_ls_pipeline.py \
    --configs_file_path "${DIR}/data_regulatome_final_configs.json" \
    --log_file_path "${DIR}/logs/sample_data_input_1.log" \
    --model_folder_path "${DIR}/model" \
    --input_folder_path "${DIR}/sample_data/input_1" \
    --output_folder_path "${DIR}/outputs/sample_data/input_1"

python run_ls_pipeline.py \
    --configs_file_path "${DIR}/data_regulatome_final_configs.json" \
    --log_file_path "${DIR}/logs/sample_data_input_2.log" \
    --model_folder_path "${DIR}/model" \
    --input_folder_path "${DIR}/sample_data/input_2" \
    --output_folder_path "${DIR}/outputs/sample_data/input_2"
