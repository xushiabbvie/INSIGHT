#!/bin/bash
inputdir="../"
gene2idfile=$inputdir"/data/gene2ind.txt"
cell2idfile=$inputdir"/data/cell2ind.txt"
drug2idfile=$inputdir"/data/drug2ind.txt"
testdatafile=$inputdir"/data/demo_test_data.txt"

mutationfile=$inputdir"/data/cell2feature.txt"
drugfile=$inputdir"/data/drug2fingerprint.txt"

modelfile1=$inputdir"/model/model_lv1_final.pt"
modelfile2=$inputdir"/model/model_lv2_final.pt"
cudaid=0

modeldir=$inputdir"/predict"
mkdir $modeldir

source activate pytorch

CMD="python -u $inputdir/code/predict_INSIGHT.py -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -test $testdatafile -model1 $modelfile1 -model2 $modelfile2 -outputdir $modeldir -cuda $cudaid -multiomics $mutationfile -fingerprint $drugfile -batchsize 10000"

echo $CMD > 'demo_predict.log'

$CMD >> demo_predict.log
