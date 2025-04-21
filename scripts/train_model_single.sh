#!/bin/bash
inputdir="../"
gene2idfile=$inputdir"/data/gene2ind.txt"
cell2idfile=$inputdir"/data/cell2ind.txt"
drug2idfile=$inputdir"/data/drug2ind.txt"
traindatafile=$inputdir"/data/demo_train_data.txt"
valdatafile=$inputdir"/data/demo_val_data.txt"
ontfile=$inputdir"/data/REACTOME_ont.txt"

mutationfile=$inputdir"/data/cell2feature.txt"
drugfile=$inputdir"/data/drug2fingerprint.txt"

cudaid=0

modeldir=$inputdir"/model"
mkdir $modeldir

source activate pytorch

CMD="python -u $inputdir/code/train_INSIGHT.py -gamma 0.001 -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -val $valdatafile -model $modeldir -cuda $cudaid -multiomics $mutationfile -fingerprint $drugfile -genotype_hiddens 8 -genotype_hiddens_gf 128 -drug_hiddens 128,128,128 -final_hiddens 128 -epoch1 100 -epoch2 200 -save -batchsize 10000"

echo $CMD > 'demo_train.log'

$CMD >> demo_train.log
