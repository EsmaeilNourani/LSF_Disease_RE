# Code for training relation extraction models
For this code, you will need to do the following steps, and code needs to run on a GPU-cluster with a slurm system.

## Preparing the system
In the first step, you need to clone the repository to your cluster drive space.
Assume you copy to your home directory `$HOME`

You must first do a git clone:
```
cd $HOME
git clone git@github.com:farmeh/RegulaTome_extraction.git
```

Then you need to download the train, devel and test data, and download the base RoBERTa-large-PM-M3-Voc model and prepare output folders using the following script:
```
cd RegulaTome_extraction/TrainRelationExtractionSystem
sh get_data.sh
```

This will download the data from [Zenodo](https://zenodo.org/api/records/10808330/files/RegulaTome-corpus.tar.gz/content) and extract it into the following directories:

- train_folder: `$HOME/RegulaTome_extraction/TrainRelationExtractionSystem/RegulaTomeCorpus/train-set`
- devel_folder: `$HOME/RegulaTome_extraction/TrainRelationExtractionSystem/RegulaTomeCorpus/devel-set`

The script will then create all necessary model and output directories:

```
model                        : $HOME/RegulaTome_extraction/TrainRelationExtractionSystem/MODEL
main output folder           : $HOME/RegulaTome_extraction/TrainRelationExtractionSystem/OUTPUTS
keeps control of gpu-jobs    : $HOME/RegulaTome_extraction/TrainRelationExtractionSystem/OUTPUTS/jobs
keeps training log files     : $HOME/RegulaTome_extraction/TrainRelationExtractionSystem/OUTPUTS/logs
keeps predictions and models : $HOME/RegulaTome_extraction/TrainRelationExtractionSystem/OUTPUTS/preds
keeps cluster log files      : $HOME/RegulaTome_extraction/TrainRelationExtractionSystem/OUTPUTS/cluster-logs
```

The script will then try to download the [RoBERTa-large-PM-M3-Voc model](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz) (pre-trained RoBERTa model on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed),
which is used by our system and extract it into the model folder: `$HOME/ComplexTome_extraction/TrainRelationExtractionSystem/MODEL/`.
In case this fails, you can manually download the pre-trained model from [here](https://github.com/facebookresearch/bio-lm/blob/main/README.md) and extract it to the model folder.

RoBERTa-large-PM-M3-Voc model is a RoBERTa model pre-trained on biomedical texts, but it is not fine-tuned to extract Complex Formation relations.
By running our training pipeline, this model will be fine-tuned on ComplexTome training data to extract Complex Formation relations from the scientific literature.

If everything goes right, then the model should be here:
- model_address: `$HOME/RegulaTome_extraction/TrainRelationExtractionSystem/MODEL/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf`

If not, make sure to download the model manually and place it correctly into that folder.

## Steps to train/finetune the model on the Puhti supercomputer

Navigate to the directory where the code for relation extraction resides. In this case: `$HOME/RegulaTome_extraction/TrainRelationExtractionSystem`.

From there you should execute `run_training_pipeline.sh` (e.g. `sh run_training_pipeline.sh`).
This code submits 4 gpu-jobs for training 4 relation extraction models, each with a different random seed index 
(basically passing a random seed index when submitting a gpu-job, e.g. `sbatch train_single_re_model.sh 1`).

This is important that you first edit `train_single_re_model.sh` file, at least to assign the following parameters correctly:
1. SBATCH --account : to assign your project account number
2. export TRANSFORMERS_CACHE= : to assign your transformer cache folder. 

Once the gpu-jobs are submitted and completed, you can check the `OUTPUTS` folder and by checking the .log files, see which model has yielded the highest f-score,
and take that model for subsequent use.


## Technical consideration on Puhti

You need to install spacy, scispacy, and the `en_core_sci_sm model`, a full spaCy pipeline for biomedical data, under your user. 

```
module purge
module load pytorch/1.12
python -m pip install --user spacy==2.3.2
python -m pip install --user scispacy==0.2.5
python -m pip install --user https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
```


