import os
import pandas as pd
import numpy as np
import random
import datetime
        
import Data_Load.Dataloader as Dataloader
import Trainer 

from sklearn.model_selection import StratifiedKFold # KFold, GridSearchCV, RandomizedSearchCV

def Experiment(args, model_type):
    
    # make a directory to save results, models
    args['total_path'] = args['save_root'] + str(args['seed'])+ "/" + args['result_dir']
    
    if not os.path.isdir(args['total_path']):
        os.makedirs(args['total_path'])
        
    # save ARGUEMENT
    with open(args['total_path'] + '/args.txt', 'a') as f:
        f.write('Preprocess Start: '+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        f.write("\n"+str(args)+"\n\n")
    
    # connect GPU/CPU
    import torch.cuda
    args['cuda'] = torch.cuda.is_available()
    # check if GPU is available, if True chooses to use it
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## fix seed
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args['cuda']:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    ## Trainer
    result=start_Training(args, model_type) # train
    return result

def start_Training(args, model_type):
    # MODEL
    model = load_model(args)

    '''
    ################################
    #     INTERNAL VALIDATION      #
    ################################
    '''
    ## Load INTERNAL Dataset
    INTERNAL_DATASET=Dataloader.init_dataset(args["internal"], args)
    
    ## dataset split
    kf = StratifiedKFold(n_splits=args['kfold'], random_state=args['seed'], shuffle=True)
    CROSS_VALIDATION_IDX=[[train, test] for train, test in kf.split(INTERNAL_DATASET.loc[:, INTERNAL_DATASET.columns != args['target']], INTERNAL_DATASET.loc[:, args['target']])]
    
    internal_result=[]
    for i, (TRAIN, TEST) in enumerate(CROSS_VALIDATION_IDX): # cross-validation
        TRAIN_LOADER, TEST_LOADER=Dataloader.preprocess(args, INTERNAL_DATASET.iloc[TRAIN, :], INTERNAL_DATASET.iloc[TEST, :], model_type)
        trainer=Trainer.Trainer(args, model, model_type)
        trainer.train(TRAIN_LOADER, i+1) 
        r=trainer.eval("valid", TEST_LOADER, i+1)
        internal_result.append(r)   
    
    return internal_result    

def load_model(args): 
    """     load model    """
    if args['model_name']=="rf":
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier(random_state=args['seed'])
    elif args['model_name']=="xgb":
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=args['max_iter'], random_state=args['seed'])
    elif args['model_name']=="lightGBM":
        from lightgbm import LGBMClassifier
        model=LGBMClassifier(n_estimators=args['max_iter'], random_state=args['seed'])
    elif args['model_name']=="svm":
        import sklearn.svm as svm
        model=svm.SVC(kernel = args["svm_kernel"], random_state=args['seed'])
    elif args['model_name']=="lr":
        from sklearn.linear_model import LogisticRegression 
        model=LogisticRegression(max_iter=args['max_iter'], random_state=args['seed'])
    elif args['model_name']=="knn":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5) 
    elif args['model_name']=="gbt":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=args['seed'], max_depth=8, learning_rate=0.05) 
    else:
        model=None
        print("Model not loaded.....")
        
    return model


def main(args, model_type):
    """     start    """
    print(f"{'*'*15} {args['dataset_name']} {'*'*15}")
    exp_type=f"{args['target']}/{args['month']}_{args['model_name']}_{args['normalize']}_{args['balance']}"

    args['result_dir']=exp_type
    
    # make directory for results
    result_path = args['save_root'] + str(args['seed']) + "/Results" 
    if not os.path.isdir(f"{result_path}/{args['target']}"):
        os.makedirs(f"{result_path}/{args['target']}")

    # start training framework
    result=pd.DataFrame(Experiment(args, model_type), columns=["Test_loss", "Test_Acc", "Test_F1", "Test_AUROC"])
    result=pd.concat([result, pd.DataFrame(result.sum()/result.shape[0]).T], axis=0)
    
    result.to_csv(f"{result_path}/{exp_type}.csv")
    
                
        