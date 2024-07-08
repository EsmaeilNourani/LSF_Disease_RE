#!/bin/bash
wget 'https://zenodo.org/records/12684263/files/LSD600.tar.gz/content' -O data.tar.gz
tar -xvf data.tar.gz -C .
rm data.tar.gz
mkdir -p LSD600Corpus
mv train LSD600Corpus/train-set
mv dev LSD600Corpus/devel-set
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
