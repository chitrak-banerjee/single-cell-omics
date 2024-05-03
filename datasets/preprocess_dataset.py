import numpy as np
import pandas as pd
import os
import re
import random
import torch
from torch.utils.data import TensorDataset

def preprocess_sinai(args):
    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True )
        os.makedirs(args.tmp_dir+'/bags', exist_ok=True)
        os.makedirs(args.tmp_dir+'/folds', exist_ok=True)
    
    dataInst = pd.read_csv(args.dataset_path)
    dataInst['case'] = dataInst['bag']
    
    # Assign into NumPy Array
    insts = np.array( dataInst.iloc[:,2] )
    bags = np.array( dataInst.iloc[:,-1] )
    cases = np.array( dataInst[['case']] )
    labels = np.array( dataInst.iloc[:,0] )
    feats = np.array( dataInst.iloc[:,3:33])

    # PyTorch Tensor
    label2bool = np.array([True if l == 'AD' else False for l in labels])
    data = TensorDataset(torch.Tensor(feats), torch.Tensor(label2bool))
    
    dataBag = dataInst[['bag', 'case', 'label']]
    print( dataBag.shape )
    
    dataBag = dataBag.drop_duplicates()
    dataBag = dataBag.rename( columns={
	    'bag': 'slide_id',
	    'case': 'case_id'
    } )
    dataBag.to_csv(args.tmp_dir+"/bag.csv", index = False)
    
    insts_in_bags = {}
    save_dir = "{}/bags".format(args.tmp_dir)
    for b in np.unique(bags): # this iterates 56 times
        insts_sub = np.where(bags == b)[0]
        feats_sub = data.tensors[0][insts_sub,:]
        insts_in_bags[b] = insts[insts_sub]
        path = "{}/{}.pt".format(save_dir, b )
        torch.save(feats_sub, path)
    print('Preprocessing Complete')
