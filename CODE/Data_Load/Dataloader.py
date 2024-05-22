import pandas as pd
import datetime
import os
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

from Utils.Normalization import Normalize
from Utils.Balancing import Balance
from Utils.Feature_selection import Feature_Selection

import shutil

def init_dataset(name, args):
    data_set=load_dataset(name) # load dataset class
    make_dataset, exist = makedataset_or_not(args['save_data'])  # fond out whether to make a new dataset or not
    
    if make_dataset:
        ## make datset
        start_time=datetime.datetime.now()
        if exist: shutil.rmtree(f"{args['save_data']}") # deleter before saved data folder
        dataset=data_set(args)
        print(f"Dataset make: {datetime.datetime.now()-start_time}")
    
    print("Dataset load!")
    dataset=pd.read_csv(f"{args['save_data']}/{name}.csv") # read {dataset_name}.csv file (= subject basic information)
    
    dataset=class_division(name, args['target'], dataset, args['month'])
    
    ## Preprocess (feature selection and rearrange dataset)
    dataset=load_preprocess(dataset, name, args)
    dataset=Feature_Selection(args["feature_select"], dataset, args['target'], args["max_iter"], args["seed"])
    args["selected_feature_name"]=dataset.columns[:-1]
    
    return dataset

def preprocess(args, train, valid, type):         
    ''' TRAIN '''
   
    TRAIN_X,TRAIN_Y=train.loc[:, train.columns != args["target"]], train.loc[:, args["target"]]
    TRAIN_X,TRAIN_Y=Normalize(args['normalize'], TRAIN_X, TRAIN_Y) ## normalize
    TRAIN_X,TRAIN_Y, WEIGHT=Balance(args["balance"], args['seed'], TRAIN_X, TRAIN_Y) ## balancing
    
    ''' VALID '''
    VALID_X,VALID_Y=valid.loc[:, valid.columns != args["target"]], valid.loc[:, args["target"]]
    VALID_X,VALID_Y=Normalize(args['normalize'], VALID_X, VALID_Y) ## normalize
    
    ''' Make data loader '''
    loaders=load_dataloader(args, TRAIN_X,TRAIN_Y, VALID_X,VALID_Y, type, WEIGHT)
    return loaders
    
    
def load_dataloader(args, TRAIN_X,TRAIN_Y, VALID_X,VALID_Y, type, WEIGHT):
    
    batch_size=TRAIN_Y.shape[0]
    valid_batch_size=VALID_Y.shape[0]

    if WEIGHT is not None:
        sampler=torch.utils.data.WeightedRandomSampler(WEIGHT, replacement=True, num_samples=batch_size) 
    else:
        sampler=None
    
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(TRAIN_X, dtype=torch.float32), torch.tensor(TRAIN_Y.tolist(), dtype=torch.float32)),
                                        batch_size=batch_size, sampler=sampler, pin_memory=True), \
           torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(VALID_X, dtype=torch.float32), torch.tensor(VALID_Y.tolist(), dtype=torch.float32)), 
                                        batch_size=valid_batch_size, shuffle=False, pin_memory=True)
           
           
def load_dataset(name):
    from Data_Load.MIMIC4_Dataset import MIMIC4_Dataset        
    dataset=MIMIC4_Dataset
    return dataset
    
    
def class_division(d_name, target, df, month):
    ## find out whether the subject is alive or dead after 'month' month after discharge
    df.loc[(0<df[target]) & (df[target]<month*30*60*24), target]=0 #  dead after 'month' month after discharge
    df.loc[df[target]>=month*30*60*24, target]=1 # alive after 'month' month after discharge
    return df
    
    
def load_preprocess(dataset, name, args):
    from Preprocess.MIMIC_preprocess import main      
    dataset=main(args, dataset)
    return dataset

def makedataset_or_not(path):
    """    find out whether to make a datset or not    """
    if not os.path.exists(f"{path}"): # dataset does not exist
        return True, 0
    
    ## make a new dataset if th made dataset is 7 days behind
    with open(path + '/args.txt', 'r') as f:
        line=f.readline()
    if (datetime.datetime.strptime(line.split()[0],'%Y%m%d')+datetime.timedelta(days=7))<datetime.datetime.now():
        return True, 1
    else:
        return False, 0
