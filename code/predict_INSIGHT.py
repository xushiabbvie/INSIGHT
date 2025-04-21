import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from INSIGHT_NN import *
import argparse
import numpy as np
import time
import pandas as pd

 
def predict_model(test_data, batch_size, cell_features, drug_features, trained_model, outfile, combo_flag, hidden_file):

	test_feature, test_label = test_data

	if os.path.isfile(trained_model):
		print("Trained model exists:" + trained_model)
		model = torch.load(trained_model, map_location='cuda:%d' % CUDA_ID) #param_file
	else:
		print("ERROR: Model does not exist.")
		sys.exit(1)

	test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

	#Test
	model.eval()
	test_predict = torch.zeros(0,0).cuda(CUDA_ID)

	pathway_list = {}

	for i, (inputdata,labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		features = build_input_vector(inputdata, cell_features, drug_features)

		cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))

		aux_out_map, _ = model(cuda_features)

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		if hidden_file is not None:	
			for term, hidden_map in aux_out_map.items():
				if i == 0:
					pathway_list[term] = hidden_map.data.cpu().numpy()
				else:
					pathway_list[term] = np.append(pathway_list[term],hidden_map.data.cpu().numpy())


	if combo_flag:
		hp = len(test_predict) // 2
		test_predict = (test_predict[:hp]+test_predict[hp:]) / 2

	np.savetxt(outfile, test_predict.cpu().numpy(),'%.4e')
	
	if hidden_file is not None:
		df_pathway = pd.DataFrame(pathway_list)
		if combo_flag:
			hp = df_pathway.shape[0] // 2
			df_pathway = (df_pathway[:hp]+df_pathway[hp:].reset_index(drop=True)) / 2

		df_pathway = df_pathway.drop(columns=df_pathway.columns[-5:])
		df_pathway.to_csv(hidden_file, index=False)


parser = argparse.ArgumentParser(description='Predict on INSIGHT model')
parser.add_argument('-combo',help='enable combo prediction mode (Optional)',action='store_true')
parser.add_argument('-test', metavar='testing_data', help='Testing dataset (Required)', type=str, required=True)
parser.add_argument('-model1', metavar='model1', help='Level 1 model used for prediction', type=str, required=True)
parser.add_argument('-model2', metavar='model2', help='Level 2 model used for prediction', type=str, required=True)
parser.add_argument('-gene2id', metavar='gene_id',help='Gene to ID mapping file (Required)', type=str, required=True)
parser.add_argument('-drug2id', metavar='drug_id',help='Drug to ID mapping file (Required)', type=str, required=True)
parser.add_argument('-cell2id', metavar='cell_id',help='Cell to ID mapping file (Required)', type=str, required=True)
parser.add_argument('-multiomics', metavar='multiomics_data', help='Multiomics data for cell lines (Required)', type=str, required=True)
parser.add_argument('-fingerprint', metavar='drug_fingerprint',help='Morgan fingerprint representation for drugs (Required)', type=str, required=True)
parser.add_argument('-outputdir', metavar='output_directory',help='Folder for trained models (default=MODEL/)', type=str, default='MODEL/')
parser.add_argument('-batchsize', metavar='batch_size',help='Batchsize (default=5000)', type=int, default=5000)
parser.add_argument('-cuda', metavar='CUDA_ID',help='Specify GPU (default=0)', type=int, default=0)
parser.add_argument('-hidden', help='enable pathway activity output to pathway_activity.txt in outputdir (Optional)',action='store_true')


# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=5)

# load input data
test_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.test, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_features = np.genfromtxt(opt.multiomics, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

#####################################

CUDA_ID = opt.cuda

### level 1 prediction
out_file1 = opt.outputdir+'/test_predicted_lv1.txt'
predict_model(test_data, opt.batchsize, cell_features, drug_features, opt.model1, out_file1, opt.combo, None)

### level 2 prediction
hidden_file = None
if opt.hidden:
	hidden_file = opt.outputdir+'/pathway_activity.txt'

out_file2 = opt.outputdir+'/test_predicted_lv2.txt'
predict_model(test_data, opt.batchsize, cell_features, drug_features, opt.model2, out_file2, opt.combo, hidden_file)

### combine prediction
out_file = opt.outputdir+'/test_predicted_final.txt'
combine_models(opt.test, out_file1, out_file2, out_file, opt.combo)





