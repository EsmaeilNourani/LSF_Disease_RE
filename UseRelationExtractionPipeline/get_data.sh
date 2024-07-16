#!/bin/bash
wget 'https://zenodo.org/records/10808330/files/combined_input_for_re.tar.gz?download=1' -O input.tar.gz
tar -xvf input.tar.gz -C ./inputs
rm input.tar.gz
