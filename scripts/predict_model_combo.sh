#!/bin/bash
inputdir="../"
gene2idfile=$inputdir"/data_combo/gene2ind.txt"
cell2idfile=$inputdir"/data_combo/cell2ind.txt"
drug2idfile=$inputdir"/data_combo/drug2ind.txt"
testdatafile=$inputdir"/data_combo/demo_test_data_combo.txt"

mutationfile=$inputdir"/data_combo/cell2feature.txt"
drugfile=$inputdir"/data_combo/drug2fingerprint.txt"

modelfile1=$inputdir"/model_combo/model_lv1_final.pt"
modelfile2=$inputdir"/model_combo/model_lv2_final.pt"
cudaid=0

modeldir=$inputdir"/predict_combo"
mkdir $modeldir

source activate pytorch

CMD="python -u $inputdir/code/predict_INSIGHT.py -combo -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -test $testdatafile -model1 $modelfile1 -model2 $modelfile2 -outputdir $modeldir -cuda $cudaid -multiomics $mutationfile -fingerprint $drugfile -batchsize 10000"

echo $CMD > 'demo_predict_combo.log'

$CMD >> demo_predict_combo.log
