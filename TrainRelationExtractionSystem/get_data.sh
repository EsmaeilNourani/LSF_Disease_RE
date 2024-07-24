#!/bin/bash
# Step 1: Download the archive
wget 'https://zenodo.org/records/12804856/files/LSD600.tar.gz' -O data.tar.gz

# Step 2: Extract the contents of the archive
tar -xvf data.tar.gz

# Step 3: Remove the downloaded archive to save space
rm data.tar.gz

# Step 4: Rename the root folder
mv LSD600 LSD600Corpus

# Step 5: Rename the subfolders
mv LSD600Corpus/train LSD600Corpus/train-set
mv LSD600Corpus/dev LSD600Corpus/devel-set
mv LSD600Corpus/test LSD600Corpus/test-set

mkdir -p MODEL
mkdir -p OUTPUTS
mkdir -p OUTPUTS/jobs
mkdir -p OUTPUTS/logs
mkdir -p OUTPUTS/preds
mkdir -p OUTPUTS/clusterlogs

wget 'https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz' -O model.tar.gz
tar -xvf model.tar.gz -C ./MODEL
rm model.tar.gz
