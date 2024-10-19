# Code for training relation extraction models
For this code, you will need to do the following steps, and code needs to run on a GPU-cluster with a slurm system.

## Preparing the system
In the first step, you need to clone the repository to your cluster drive space.
Assume you copy to your home directory `$HOME`

You must first do a git clone:
```
cd $HOME
git clone https://github.com/EsmaeilNourani/LSF_Disease_RE.git

```

Then you need to download the train, devel and test data, and download the base RoBERTa-large-PM-M3-Voc model and prepare output folders using the following script:
```
cd LSF_Disease_RE/TrainRelationExtractionSystem
sh get_data.sh
```

This will download the data from [Zenodo](https://zenodo.org/records/13952449/files/LSD600.tar.gz) and extract it into the following directories:


- train_folder: `$HOME/LSF_Disease_RE/TrainRelationExtractionSystem/LSD600Corpus/train-set`
- devel_folder: `$HOME/LSF_Disease_RE/TrainRelationExtractionSystem/LSD600Corpus/devel-set`

The script will then create all necessary model and output directories:

```
model                        : $HOME/LSF_Disease_RE/TrainRelationExtractionSystem/MODEL
main output folder           : $HOME/LSF_Disease_RE/TrainRelationExtractionSystem/OUTPUTS
keeps control of gpu-jobs    : $HOME/LSF_Disease_RE/TrainRelationExtractionSystem/OUTPUTS/jobs
keeps training log files     : $HOME/LSF_Disease_RE/TrainRelationExtractionSystem/OUTPUTS/logs
keeps predictions and models : $HOME/LSF_Disease_RE/TrainRelationExtractionSystem/OUTPUTS/preds
keeps cluster log files      : $HOME/LSF_Disease_RE/TrainRelationExtractionSystem/OUTPUTS/cluster-logs
```

The script will then try to download the [RoBERTa-large-PM-M3-Voc model](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz) (pre-trained RoBERTa model on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed),
which is used by our system and extract it into the model folder: `$HOME/ComplexTome_extraction/TrainRelationExtractionSystem/MODEL/`.
In case this fails, you can manually download the pre-trained model from [here](https://github.com/facebookresearch/bio-lm/blob/main/README.md) and extract it to the model folder.

RoBERTa-large-PM-M3-Voc model is a RoBERTa model pre-trained on biomedical texts, but it is not fine-tuned to extract Complex Formation relations.
By running our training pipeline, this model will be fine-tuned on ComplexTome training data to extract Complex Formation relations from the scientific literature.

If everything goes right, then the model should be here:
- model_address: `$HOME/LSF_Disease_RE/TrainRelationExtractionSystem/MODEL/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf`

If not, make sure to download the model manually and place it correctly into that folder.

## Steps to train/finetune the model on the C2 supercomputer

Navigate to the directory where the code for relation extraction resides. In this case: `$HOME/LSF_Disease_RE/TrainRelationExtractionSystem`.


From there you should execute train_single_re_model.sh (e.g., `qsub train_single_re_model.sh`). This code submits a GPU job for training a relation extraction model. You can change the random seed index in the script before running it. To train multiple models with different random seed indices, modify the random_seed_index variable in train_single_re_model.sh for each run.

* Best model will be saved to `/LSF_Disease_RE/TrainRelationExtractionSystem/OUTPUTS/` for later use.

This is important that you first edit `train_single_re_model.sh` file, at least to assign the following parameters correctly:
1. PBS -W group_list= : to assign your project group name
2. export TRANSFORMERS_CACHE= : to assign your transformer cache folder. 

Once the gpu-jobs are submitted and completed, you can check the `OUTPUTS` folder and by checking the .log files, see which model has yielded the highest f-score,
and take that model for subsequent use.

## Considerations about using C2
* You may consider loading the existing modules on C2 for Tensorflow and Pytorch instead of installing them
```
module purge
module load tools computerome_utils/2.0
module load anaconda3/2022.10
module load pytorch/1.12.1
module load tensorflow/2.9.1
```

* Still you need to install the following libraries
```
#run the next 4 commands only once, then you have them in your user.
python -m pip install --user spacy==2.3.2
python -m pip install --user scispacy==0.2.5
python -m pip install --user https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
python -m pip install --user transformers==4.20.1
```

