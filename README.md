# Lifestyle-Disease Relation Extraction

This is a forked [repository](https://github.com/farmeh/RegulaTome_extraction) ([paper](https://www.biorxiv.org/content/10.1101/2024.04.30.591824v1)) adapted for training on the LifeStyle-Disease relation corpus (LSD600) and to use it for Disease-Lifestyle relation extraction.



There are two main folders:
1. Code for training the torch-based relation extraction system on LSD600 training data.
2. Code for extraction of Lifestyle-Disease relations using the trained relation extraction system.




### Environment setup:
This code is tested with Python 3.9 installed with conda and the packages from requirements.txt installed in that environment. 

```
conda create -n RE python=3.9
conda activate RE
pip install -r requirements.txt
```

## Considerations about using C2 Supercomputer
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

This will download the data from [Zenodo](https://zenodo.org/records/12684263/files/LSD600.tar.gz) and extract it into the following directories:


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




# Code for running prediction with the best model
This contains all the scripts to run the relation extraction pipeline on the test data or any documents of your choice. 

**IMPORTANT:** Please note that the pipeline is only capable of extracting relations between different entities that are **aleady detected** in the texts with an NER system. 
In other words, the pipeline does not include any NER system as an internal component. 

However, 
- To run the system on the test data, we have already ran an NER system, and the files are provided in BRAT standoff format (.txt and .ann). 
  Please check the structure of this folder to get a better understanding of how inputs should look like for the pipeline.

- To run the system on your documents of your choice, you need to first run an NER system to detect entities (e.g. Lifestyle Factors),
  and provide the documents in BRAT standoff formant and put them in the input folders. 
  To get a better understanding of the input folder structure, you can check the [test data](https://github.com/EsmaeilNourani/LSF_Disease_RE/tree/main/UseRelationExtractionPipeline//sample_data/LSD600Corpus/test-set) folder. 


To run the pipeline you have to follow below instructions:

1. If you haven't done so already, you will need to clone the repository to your cluster drive space. Assume you copy to your home directory $HOME
You must first do a git clone:
    ```
    cd $HOME
    git clone https://github.com/EsmaeilNourani/LSF_Disease_RE.git
    ```

2. Then you will need to run the following script to create all necessary folders:
    ```
    cd LSF_Disease_RE/UseRelationExtractionPipeline
    bash make_folders.sh
    ```
    This script creates the following necessary folders: 
    ```
     model   : $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model
     inputs  : $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/inputs
     outputs : $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/outputs
     jobs    : $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/jobs
     logs    : $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/logs
     cluster_logs : $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/cluster_logs
    ```

3. Then you have two alternative options to proceed (choose only one): 
    - 3.1: Use the relation extraction model we have already trained:
    To do so, please execute the following script which downloads the model from [Zenodo](https://zenodo.org/records/12684263/files/best-model-LSF-undir-RoBERTa-5e-6-180-75.tar.gz?download=1), 
    and extracts it into the `model` folder. 
    
       ```
       bash get_model.sh
       ```
       once this is done, you should have the following files in the `model` folder: 
       ```
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/added_tokens.json
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/config.json
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/info.json
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/merges.txt
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/pytorch_model.bin
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/special_tokens_map.json
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/tokenizer.json
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/tokenizer_config.json
       $HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/vocab.json
       ```
        
    - 3.2: If you have trained a model yourself using the code in the `TrainRelationExtractionSystem` directory, one can proceed with running the relation extraction system with that model.
      You need to place all model files (see above) into the `model` folder : `HOME/LSF_Disease_RE/UseRelationExtractionPipeline/model/`

4. Finally, you can proceed with running the pipeline on sample_data, or entire literature data, or any documents of your choice. 


## 1. Prediction on test data
In order to run the relation extraction system on [test data](https://github.com/EsmaeilNourani/LSF_Disease_RE/tree/main/UseRelationExtractionPipeline//sample_data/LSD600Corpus/test-set), you need to first edit `run_system_predict_sample_data.sh` script and add necessary information (e.g your account information on the cluster, etc), 
and then run it with the following command on your cluster gpu machine:
```
qsub run_system_predict_sample_data.sh
```

This uses the data in the `sample_data` directory and the model you have put in the `model` folder. 
Once the execution is complete, you can check `outputs`, `logs`, and `cluster_logs` folders.


###Important note: 
Please note that this script is based on the qsub command and the PBS system. If you are plannig to run it on a different cluster system, you have to edit the files based on your own cluster environment commands.

## 2. Run the system on your own files
For this, you will need to first run an NER system to detect entities and provide the output in BRAT format.
Then distribute your BRAT files (.ann and .txt) into input folder.
Then you have to edit `run_system_predict_sample_data.sh` script and add necessary information (e.g your account information on the cluster, etc) folder. 
You can check the structure of files inside the sample_data folder to get a better understanding of how files should be provided to the system as input. 
Finally, you can follow the same instructions given for running the system on sample data to run the system on your own files.



