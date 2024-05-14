#!/bin/bash
### run this code to train four different relation extraction models.
### each line submits a gpu-job to train a model with a different random_seed
sbatch train_single_re_model.sh 0
sbatch train_single_re_model.sh 1
sbatch train_single_re_model.sh 2
sbatch train_single_re_model.sh 3
