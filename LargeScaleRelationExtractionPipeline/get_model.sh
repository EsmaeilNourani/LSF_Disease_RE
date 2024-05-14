#!/bin/bash
wget 'https://zenodo.org/records/10808330/files/relation_extraction_multi-label-best_model.tar.gz?download=1' -O model.tar.gz
tar -xvf model.tar.gz -C ./model
rm model.tar.gz
mv model/relation_extraction_multi-label-best_model/* model/
cd model
rm -rf relation_extraction_multi-label-best_model
