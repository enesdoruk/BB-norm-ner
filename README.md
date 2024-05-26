# BB Entity Normalization and Entity Recognition 

Biology and bioinformatics projects produce huge amounts of heterogeneous information about the microbial strains that have been experimentally identified in a given environment (habitat), and theirs properties (phenotype). These projects include applied microbiology domain (food safety), health sciences and waste processing. Knowledge about microbial diversity is critical for studying in depth the microbiome, the interaction mechanisms of bacteria with their environment from genetic, phylogenetic and ecology perspectives. A large part of the information is expressed in free text in large sets of scientific papers, web pages or databases. Thus, automatic systems are needed to extract the relevant information. The BB task aims to encourage the development of such systems.

**BB-norm:** Normalization of Microorganism, Habitat and Phenotype entities with
NCBI Taxonomy taxa (for the former) and OntoBiotope habitat concepts (for the
last two). Entity annotations are provided.

**BB-norm+ner:** Recognition of Microorganism, Habitat and Phenotype entities and
normalization with NCBI Taxonomy taxa and OntoBiotope habitat concepts.

## Setup 

Create conda environmenet 

- ```conda env create -f environment.yml```

-  ```conda activate bbnorm-ner```

## HyperParameters
You can change thsese hypaerparameters in norm.py and norm_ner.py

*   Learnin Rate
*   Number Of Epochs
*   Max Legth 
*   Embedding Size


## Train and Test 
* For BB-norm 
    - ```python norm.py```
    - ```python norm_ner.py```

