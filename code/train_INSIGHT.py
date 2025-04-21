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


# build mask: matrix (nrows = number of relevant gene set, ncols = number all genes)
# elements of matrix are 1 if the corresponding gene is one of the relevant genes
def create_term_mask(term_direct_gene_map, gene_dim):

	term_mask_map = {}

	for term, gene_set in term_direct_gene_map.items():

		mask = torch.zeros(len(gene_set), gene_dim)

		for i, gene_id in enumerate(gene_set):
			mask[i, gene_id] = 1

		mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))

		term_mask_map[term] = mask_gpu

	return term_mask_map

 
def train_model(root, term_size_map, term_direct_gene_map, dG, train_data, gene_dim, drug_dim, model_save_folder, train_epochs, batch_size, learning_rate, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features, gamma, num_hiddens_gf, pretrained_model, save_flag, levels):

	epoch_start_time = time.time()
	best_epoch = 0
	max_corr = 0

	# dcell neural network
	model = drugcell_nn(term_size_map, term_direct_gene_map, dG, gene_dim, drug_dim, root, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final,num_hiddens_gf)

	train_feature, train_label, val_feature, val_label = train_data

	train_label_gpu = torch.autograd.Variable(train_label.cuda(CUDA_ID))
	val_label_gpu = torch.autograd.Variable(val_label.cuda(CUDA_ID))

	model.cuda(CUDA_ID)

	if os.path.isfile(pretrained_model):
		print("Pre-trained model exists:" + pretrained_model)
		model = torch.load(pretrained_model, map_location='cuda:%d' % CUDA_ID) #param_file
	else:
		print("Pre-trained model does not exist, start training a new model.")

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
	term_mask_map = create_term_mask(model.term_direct_gene_map, gene_dim)

	for name, param in model.named_parameters():
		if '_direct_gene_layer.weight' not in name:
			continue
		param.requires_grad = False
		term_name = name.split('_')[0]
		param.data.fill_(1)
		param.data = torch.mul(param.data, term_mask_map[term_name])

	optimizer.zero_grad()

	train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
	val_loader = du.DataLoader(du.TensorDataset(val_feature,val_label), batch_size=batch_size, shuffle=False)

	best_model = model

	for epoch in range(train_epochs):

		#Train
		model.train()
		train_predict = torch.zeros(0,0).cuda(CUDA_ID)

		epoch_loss = 0

		for i, (inputdata, labels) in enumerate(train_loader):
			# Convert torch tensor to Variable
			features = build_input_vector(inputdata, cell_features, drug_features)

			cuda_features = torch.autograd.Variable(features.cuda(CUDA_ID))
			cuda_labels = torch.autograd.Variable(labels.cuda(CUDA_ID))

			# Forward + Backward + Optimize
			optimizer.zero_grad()  # zero the gradient buffer

			# Here term_NN_out_map is a dictionary 
			aux_out_map, _ = model(cuda_features)

			if train_predict.size()[0] == 0:
				train_predict = aux_out_map['final'].data
			else:
				train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)

			total_loss = 0	
			for name, output in aux_out_map.items():
				loss = nn.MSELoss()
				if name == 'final':
					total_loss += loss(output, cuda_labels)
				else: # change 0.2 to smaller one for big terms
					total_loss += gamma * loss(output, cuda_labels)
			epoch_loss += total_loss
			total_loss.backward()

			for name, param in model.named_parameters():
				if '_direct_gene_layer.weight' not in name:
					continue
				term_name = name.split('_')[0]

			optimizer.step()

		train_corr = pearson_corr(train_predict, train_label_gpu)

		np.savetxt(model_save_folder+'/train_'+levels+'_ep'+str(epoch)+'.txt', train_predict.cpu().numpy(),'%.4e')

		###
		model.eval()
		
		val_predict = torch.zeros(0,0).cuda(CUDA_ID)

		for i, (inputdata, labels) in enumerate(val_loader):
			# Convert torch tensor to Variable
			features = build_input_vector(inputdata, cell_features, drug_features)
			cuda_features = Variable(features.cuda(CUDA_ID))

			aux_out_map, _ = model(cuda_features)

			if val_predict.size()[0] == 0:
				val_predict = aux_out_map['final'].data
			else:
				val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)

		val_corr = pearson_corr(val_predict, val_label_gpu)
		np.savetxt(model_save_folder+'/val_'+levels+'_ep'+str(epoch)+'.txt', val_predict.cpu().numpy(),'%.4e')

		epoch_end_time = time.time()
		print("epoch\t%d\tcuda_id\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\tepoch_loss\t%.6f\telapsed_time\t%s" % (epoch, CUDA_ID, train_corr, val_corr, epoch_loss, epoch_end_time-epoch_start_time))
		epoch_start_time = epoch_end_time

		if epoch > 50:
			if val_corr >= max_corr:
				max_corr = val_corr
				best_epoch = epoch
				best_model = model
			if save_flag:
				torch.save(model, model_save_folder + '/model_'+levels+'_ep'+str(epoch)+'.pt')

	torch.save(best_model, model_save_folder + '/model_'+levels+'_final.pt')	
	print("Best performed model (epoch)\t%d" % best_epoch)



parser = argparse.ArgumentParser(description='Train INSIGHT model')
parser.add_argument('-onto', metavar='ontology',help='Ontology file used to guide the neural network (Required)', type=str, required=True)
parser.add_argument('-train', metavar='training_data', help='Training dataset (Required)', type=str, required=True)
parser.add_argument('-val', metavar='validation_data',help='Validation dataset (Required)', type=str, required=True)
parser.add_argument('-gene2id', metavar='gene_id',help='Gene to ID mapping file (Required)', type=str, required=True)
parser.add_argument('-drug2id', metavar='drug_id',help='Drug to ID mapping file (Required)', type=str, required=True)
parser.add_argument('-cell2id', metavar='cell_id',help='Cell to ID mapping file (Required)', type=str, required=True)
parser.add_argument('-multiomics', metavar='multiomics_data', help='Multiomics data for cell lines (Required)', type=str, required=True)
parser.add_argument('-fingerprint', metavar='drug_fingerprint',help='Morgan fingerprint representation for drugs (Required)', type=str, required=True)
parser.add_argument('-epoch1', metavar='training_epoch_level1',help='Training epochs for training level1 model (default=100)', type=int, default=100)
parser.add_argument('-epoch2', metavar='training_epoch_level2',help='Training epochs for training level2 model (default=200)', type=int, default=200)
parser.add_argument('-lr', metavar='leaning_rate',help='Learning rate (default=0.001)', type=float, default=0.001)
parser.add_argument('-batchsize', metavar='batch_size',help='Batchsize (default=5000)', type=int, default=5000)
parser.add_argument('-modeldir', metavar='model_directory',help='Folder for trained models (default=MODEL/)', type=str, default='MODEL/')
parser.add_argument('-cuda', metavar='CUDA_ID',help='Specify GPU (default=0)', type=int, default=0)
parser.add_argument('-gamma', metavar='gamma',help='Gamma parameter for interpretability (default=0.001)', type=float, default=0.001)
parser.add_argument('-genotype_hiddens', metavar='num_hidden',help='Mapping for the number of neurons in each term in genotype parts (default=8)', type=int, default=8)
parser.add_argument('-genotype_hiddens_gf', metavar='num_hidden_top',help='Mapping for the number of neurons in top genotype parts  (default=128)', type=int, default=128)
parser.add_argument('-drug_hiddens' , metavar='num_drugs_hidden',help='Mapping for the number of neurons in each layer (default=128,128,128)', type=str, default='128,128,128')
parser.add_argument('-final_hiddens', metavar='num_MLP',help='The number of neurons in each of the MLP layer (default=128)', type=int, default=128)
parser.add_argument('-pretrained_model1', metavar='pre-trained_model1', help='Pre-trained baseline level 1 model', type=str, default='')
parser.add_argument('-pretrained_model2', metavar='pre-trained_model2', help='Pre-trained baseline level 2 model', type=str, default='')
parser.add_argument('-save',help='Enable to save model at each epoch',action='store_true')


# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=5)

# load input data
train_data, cell2id_mapping, drug2id_mapping = prepare_train_data(opt.train, opt.val, opt.cell2id, opt.drug2id)
gene2id_mapping = load_mapping(opt.gene2id)

# load cell/drug features
cell_features = np.genfromtxt(opt.multiomics, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)
drug_dim = len(drug_features[0,:])

# load ontology
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping, True)

# load the number of hiddens #######
num_hiddens_genotype = opt.genotype_hiddens

num_hiddens_drug = list(map(int, opt.drug_hiddens.split(',')))

num_hiddens_final = opt.final_hiddens
num_hiddens_gf = opt.genotype_hiddens_gf
#####################################

CUDA_ID = opt.cuda

#### training level-one model
levels="lv1"
train_model(root, term_size_map, term_direct_gene_map, dG, train_data, num_genes, drug_dim, opt.modeldir, opt.epoch1, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features, opt.gamma, num_hiddens_gf, opt.pretrained_model1, opt.save, levels)

#### training level-two model
levels="lv2"
mean0_data = opt.modeldir + "/mean0_data.txt"
create_mean0_data(opt.train,mean0_data)
train_mean0_data, cell2id_mapping, drug2id_mapping = prepare_train_data(mean0_data, opt.val, opt.cell2id, opt.drug2id)
dG, root, term_size_map, term_direct_gene_map = load_ontology(opt.onto, gene2id_mapping,False)
train_model(root, term_size_map, term_direct_gene_map, dG, train_mean0_data, num_genes, drug_dim, opt.modeldir, opt.epoch2, opt.batchsize, opt.lr, num_hiddens_genotype, num_hiddens_drug, num_hiddens_final, cell_features, drug_features, opt.gamma, num_hiddens_gf, opt.pretrained_model2, opt.save, levels)





