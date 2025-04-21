# INSIGHT: In Silico Drug Screening Platform using Interpretable Deep Learning Network
INSIGHT is an interpretable deep learning model to predict drug response on cancer cell lines.

## Setting up environment
The environment configuration is included in environment.yml. The environment can be created by conda:   

        conda env create -f environment.yml
        
The environment can be activated by
        
        source activate pytorch
        
## Training and running INSIGHT single mode
Required input files, demo data is available [here](https://figshare.com/s/acc77bf17cc8e0d4167b):
1. Multiomics data: _gene2ind.txt_, _cell2ind.txt_, _cell2feature.txt_
    * _gene2ind.txt_: A tab-delimited file with the first column as the index and the second column as the gene ID. The gene data encompasses both expression and mutation information. Gene IDs associated with expression begin with EXP, while those related to mutation data start with MUT.
    * _cell2ind.txt_: A tab-delimited file with the first column as the index and the second column as the cell line ID.
    * _cell2feature.txt_: A comma-delimited file where each row represents a cell line and each column represents a gene feature. The column headers should correspond to those in the _gene2ind.txt_ file, and the rows should align with the entries in the _cell2ind.txt_ file.
2. Drug feature files: _drug2ind.txt_, _drug2fingerprints.txt_
    * _drug2ind.txt_: A tab-delimited file with the first column as index and the 2nd column as drug ID.
    * _drug2fingerprint.txt_: A comma-delimited file with each row representing the infomax fingerprint of one drug. The rows should align with _drug2ind.txt_. The infomax fingerprint can be generated from [here](https://github.com/NetPharMedGroup/publication_fingerprint/).
3. Drug sensitivity data file: _train_data.txt_, _val_data.txt_, _test_data.txt_
    * _train_data.txt_: Training data. A tab-delimited file where the first column contains the cell line ID, the second column contains the drug ID, and the third column contains the drug sensitivity value (AUC from PRISM).
    * _val_data.txt_: Validation data used to control overfitting. Same format as the training data.
4. Pathway ontology data: _REACTOME_ontology.txt_
    * A tab-delimited file provides the REACTOME pathway annotations, featuring two types of connections. The connections between pathways are depicted by two pathway IDs, with the parent pathway in the first column and the child pathway in the second, marked by the tag 'default' to indicate pathway connections. The other type of connection represents the genes encompassed within the pathways; this is indicated by a pathway ID and a gene ID, marked by the tag 'gene' to signify pathway-gene connections. Below shows an example of pathway annotations:

```
        R-HSA-9658195   R-HSA-9664424   default
        R-HSA-9658195   R-HSA-9664433   default
        R-HSA-9659379   R-HSA-9662360   default
        R-HSA-9659379   R-HSA-9662361   default
        R-HSA-9659787   R-HSA-9661069   default
        R-HSA-6782315   EXP.LAGE3       gene
        R-HSA-6782315   EXP.TRMT9B      gene
        R-HSA-6784531   EXP.CPSF1       gene
        R-HSA-6784531   EXP.DDX1        gene
        R-HSA-6784531   EXP.NUP210      gene
        R-HSA-164843    MUT.LIG4        gene
        R-HSA-73843     MUT.PRPS1L1     gene
        R-HSA-1971475   MUT.AGRN        gene
        R-HSA-1971475   MUT.B3GAT1      gene
        R-HSA-1971475   MUT.B3GAT2      gene
```

We have provided demo data in the 'demo' folder for training the model. To train the model, execute the _train_model_single.sh_ script. Once training is complete, the model can be used to predict drug response on test data by running the _test_model_single.sh_ script.

## Training and running INSIGHT combo mode
In combo mode, the required data remains the same as in single mode, with the only difference being the format of the input data. Demo data is available [here](https://figshare.com/s/3b3bd4bfaeeee55bc640). To represent the fingerprint of a drug combination, the fingerprints of two drugs are appended together, requiring each data point to be trained twice using different orders of appending. The combo ID is formed by concatenating two drug IDs with a dot separator. Both methods of appending fingerprints should be documented in the drug2ind.txt and drug2fingerprint.txt files. For instance, the training data for drug A and drug B would be:

        cellX        A.B        0.5
        cellY        A.B        0.6
        cellX        B.A        0.5
        cellY        B.A        0.6

Demo data for training and testing the combo model is available in the 'demo_combo' folder. To train the model, execute the `train_model_combo.sh` script. Upon completing the training, the model can predict drug responses on test data by running the `test_model_combo.sh` script. For testing combo mode data, ensure the data is stacked with the same configuration and reverse the order of drugs. For example, to test the drug combinations A-B and C-D on cell lines X, Y, and Z, the test data should be:

        cellX        A.B
        cellY        A.B
        cellZ        A.B
        cellX        C.D
        cellY        C.D
        cellZ        C.D
        cellX        B.A
        cellY        B.A
        cellZ        B.A
        cellX        D.C
        cellY        D.C
        cellZ        D.C
        
