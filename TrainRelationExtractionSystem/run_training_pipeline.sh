#!/bin/bash
### run this code to train four different relation extraction models.
### each line submits a gpu-job to train a model with a different random_seed
qsub train_single_re_model.sh 0
qsub train_single_re_model.sh 1
qsub train_single_re_model.sh 2
qsub train_single_re_model.sh 3
