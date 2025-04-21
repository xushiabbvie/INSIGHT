#!/bin/bash
inputdir="../"
gene2idfile=$inputdir"/data_combo/gene2ind.txt"
cell2idfile=$inputdir"/data_combo/cell2ind.txt"
drug2idfile=$inputdir"/data_combo/drug2ind.txt"
traindatafile=$inputdir"/data_combo/demo_train_data_combo.txt"
valdatafile=$inputdir"/data_combo/demo_val_data_combo.txt"
ontfile=$inputdir"/data_combo/REACTOME_ont.txt"

mutationfile=$inputdir"/data_combo/cell2feature.txt"
drugfile=$inputdir"/data_combo/drug2fingerprint.txt"

cudaid=0

modeldir=$inputdir"/model_combo"
mkdir $modeldir

source activate pytorch

CMD="python -u $inputdir/code/train_INSIGHT.py -gamma 0.001 -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -val $valdatafile -model $modeldir -cuda $cudaid -multiomics $mutationfile -fingerprint $drugfile -genotype_hiddens 8 -genotype_hiddens_gf 128 -drug_hiddens 128,128,128 -final_hiddens 128 -epoch1 100 -epoch2 200 -save -batchsize 10000"

echo $CMD > 'demo_train_combo.log'

$CMD >> demo_train_combo.log
