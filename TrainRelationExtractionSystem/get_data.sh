#!/bin/bash
wget 'https://zenodo.org/api/records/10808330/files/RegulaTome-corpus.tar.gz/content' -O data.tar.gz
tar -xvf data.tar.gz -C .
rm data.tar.gz
mkdir -p RegulaTomeCorpus
mv train RegulaTomeCorpus/train-set
mv devel RegulaTomeCorpus/devel-set
rm -rf test
mkdir -p MODEL
mkdir -p OUTPUTS
mkdir -p OUTPUTS/jobs
mkdir -p OUTPUTS/logs
mkdir -p OUTPUTS/preds
mkdir -p OUTPUTS/clusterlogs

wget 'https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz' -O model.tar.gz
tar -xvf model.tar.gz -C ./MODEL
rm model.tar.gz
