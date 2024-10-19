#!/bin/bash
wget 'https://zenodo.org/records/13952449/files/best-model-LSF-undir-RoBERTa-5e-6-180-75.tar.gz?download=1' -O model.tar.gz
tar -xvf model.tar.gz -C ./model
rm model.tar.gz
mv model/model/* model/
cd model
rm -rf model
