# Lifestyle-Disease Relation Extraction

This is a forked [repository](https://github.com/farmeh/RegulaTome_extraction) ([paper](https://www.biorxiv.org/content/10.1101/2024.04.30.591824v1)) with minor updates to adapt the system for training on the LifeStyle-Disease relation corpus (LSD600) and to use it for Disease-Lifestyle relation extraction.



There are two main folders:
1. Code for training the torch-based relation extraction system on RegulaTome training data.
2. Code for large-scale extraction of biomedical relations from the literature, using the trained relation extraction system.




### Environment setup:
This code is tested with Python 3.9 installed with conda and the packages from requirements.txt installed in that environment. 

```
conda create -n RE python=3.9
conda activate RE
pip install -r requirements.txt
```

