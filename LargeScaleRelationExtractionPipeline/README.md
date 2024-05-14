# Code for running prediction with the best model
This contains all the scripts to run the large-scale relation extraction pipeline, 
eiher on (1) the [sample data](https://github.com/farmeh/RegulaTome_extraction/tree/main/LargeScaleRelationExtractionPipeline/sample_data), 
or (2) whole biomedical literature (PubMed abstracts and PMCOA as we have prepared them on [Zenodo](https://zenodo.org/records/10808330/files/combined_input_for_re.tar.gz?download=1)), 
or (3) any documents of your choice. 

**IMPORTANT:** Please note that the pipeline is only capable of extracting relations between different entities that are **aleady detected** in the texts with an NER system. 
In other words, the pipeline does not include any NER system as an internal component. 

However, 
- To run the system on the [sample data](https://github.com/farmeh/RegulaTome_extraction/tree/main/LargeScaleRelationExtractionPipeline/sample_data),
  we have already ran an NER system, and the files are provided in BRAT standoff format (.txt and .ann) inside .tar.gz files. 
  Please check the structure of this folder to get a better understanding of how inputs should look like for the pipeline.

- To run the system on whole biomedical literature, we have also ran an NER system on entire literature, and prepared the inputs in the proper format, 
  and placed them on [Zenodo](https://zenodo.org/records/10808330/files/combined_input_for_re.tar.gz?download=1). 

- To run the system on your documents of your choice, you need to first run an NER system to detect biomedical entities (e.g. Protein),
  and provide the documents in BRAT standoff formant and place them into one or more .tar.gz file, and distribute your tar.gz files into one or more input folders. 
  To get a better understanding of the input folder structure, you can check the [sample data](https://github.com/farmeh/RegulaTome_extraction/tree/main/LargeScaleRelationExtractionPipeline/sample_data) folder. 

To run the pipeline you have to follow below instructions:

1. If you haven't done so already, you will need to clone the repository to your cluster drive space. Assume you copy to your home directory $HOME
You must first do a git clone:
    ```
    cd $HOME
    git clone git@github.com:farmeh/RegulaTome_extraction.git
    ```

2. Then you will need to run the following script to create all necessary folders:
    ```
    cd RegulaTome_extraction/LargeScaleRelationExtractionPipeline
    bash make_folders.sh
    ```
    This script creates the following necessary folders: 
    ```
     model   : $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model
     inputs  : $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/inputs
     outputs : $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/outputs
     jobs    : $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/jobs
     logs    : $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/logs
     cluster_logs : $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/cluster_logs
    ```

3. Then you have two alternative options to proceed (choose only one): 
    - 3.1: Use the relation extraction model we have already trained:
    To do so, please execute the following script which downloads the model from [Zenodo](https://zenodo.org/records/10808330/files/relation_extraction_multi-label-best_model.tar.gz?download=1), 
    and extracts it into the `model` folder. 
       ```
       bash get_model.sh
       ```
       once this is done, you should have the following files in the `model` folder: 
       ```
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/added_tokens.json
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/config.json
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/info.json
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/merges.txt
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/pytorch_model.bin
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/special_tokens_map.json
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/tokenizer.json
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/tokenizer_config.json
       $HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/vocab.json
       ```
        
    - 3.2: If you have trained a model yourself using the code in the `TrainRelationExtractionSystem` directory, one can proceed with running the relation extraction system with that model.
      You need to place all model files (see above) into the `model` folder : `HOME/RegulaTome_extraction/LargeScaleRelationExtractionPipeline/model/`

4. Finally, you can proceed with running the pipeline on sample_data, or entire literature data, or any documents of your choice. 


## 1. Prediction on sample data
In order to run the relation extraction system on [sample data](https://github.com/farmeh/RegulaTome_extraction/tree/main/LargeScaleRelationExtractionPipeline/sample_data) data, you need to first edit `run_system_predict_sample_data.sh` script and add necessary information (e.g your account information on the cluster, etc), 
and then run it with the following command on your cluster gpu machine:
```
sbatch run_system_predict_sample_data.sh
```

This uses the data in the `sample_data` directory and the model you have put in the `model` folder. 
Once the execution is complete, you can check `outputs`, `logs`, and `cluster_logs` folders.

## 2. Large scale run on the entire literature
To replicate the large scale run, i.e running the system on the entire literature you first need to get all the data from Zenodo:
```
get_data.sh
```
Once it is done, you need to execute the following script. 
We recommend running the shell scripts that submits the jobs inside a screen, since it goes through all input directories and starts a gpu job per directory: 
```
screen -S mygrid
bash run_system_predict_entire_literature_screen.sh
```

To detach from the screen press `Ctrl+A+D` and to resume the screen `screen -r mygrid`. 
Finally, to completely terminate the screen make sure you are connected in the same login node in which you have started the screen and press `Ctrl+D`.

###Important note: 
Please first edit `puhti_single.sh` file and add all necessary information (e.g account information, TRANSFORMERS_CACHE folder path, etc). 
Also, please note that this script is written for any slurm-based cluster system. 
If you are plannig to run it on a different cluster system, you have to edit the files based on your own cluster environment commands.

## 3. Run the system on your own files
For this, you will need to first run an NER system to detect entities and provide the output in BRAT format.
Then distribute your BRAT files (.ann and .txt) into one/more .tar.gz files, and distribute your .tar.gz files into one or more input folder.
You can basically add more files into the [sample data](https://github.com/farmeh/RegulaTome_extraction/tree/main/LargeScaleRelationExtractionPipeline/sample_data) folder.
Then you have to edit `run_system_predict_sample_data.sh` script and add necessary information (e.g your account information on the cluster, etc) folder. 
You can check the structure of files inside the sample_data folder to get a better understanding of how files should be provided to the system as input. 
Finally, you can follow the same instructions given for running the system on sample data to run the system on your own files.

## Technical consideration on Puhti (gpu cluster machine)

You need to install spacy, scispacy, and the `en_core_sci_sm` model, a full spaCy pipeline for biomedical data, under your user.

```
module purge
module load pytorch
python -m pip install --user spacy==2.3.2
python -m pip install --user scispacy==0.2.5
python -m pip install --user https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.5/en_core_sci_sm-0.2.5.tar.gz
python -m pip install --user torch==1.9.1
```