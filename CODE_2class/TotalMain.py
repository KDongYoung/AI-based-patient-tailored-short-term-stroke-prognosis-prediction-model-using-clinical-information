import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

import Main

import argparse
import os

if __name__ == '__main__':
    """ Experiment Setting """ 
    # ARGUMENT
    parser = argparse.ArgumentParser(description='Stroke analysis')
    parser.add_argument('--data_root', default='DATASET/mimic4_2.2/', help="name of the data folder") # /opt/workspace/Code_Stroke/dataset//
    parser.add_argument('--run_code_folder', default='')
    parser.add_argument('--save_data', default='pre_MIMIC/', help="name of the data folder") # DATASET_DIR/
    parser.add_argument('--save_root', default='MODEL_SAVE_DIR_2/', help="where to save the models and tensorboard records") # MODEL_SAVE_DIR
    parser.add_argument('--result_dir', default="", help="save folder name") 
    parser.add_argument('--total_path', default="", help='total result path')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda')
    parser.add_argument('--cuda_num', type=int, default=0, help='cuda number')
    parser.add_argument('--device', default="", help='device')
    parser.add_argument('--disease_name', default='Stroke')
    
    parser.add_argument('--internal', default="mimic4", help='num classes')
    parser.add_argument('--n_classes', type=int, default=0, help='num classes')

    ## MIMIC-IV
    parser.add_argument('--name', default=[], help='name of files used for analysis')
    parser.add_argument('--over_age', default=True)
    parser.add_argument('--over_24_icustay', default=True)
    parser.add_argument('--over_height_over_weight', default=True)
    parser.add_argument('--missing_value', default=['gender', 'anchor_age', 'los', 'height', 'weight', 'blood_pressure_max', 'blood_pressure_min'], 
                        help='columns that needs to be considered while fill na, not just fillna(0)')
    parser.add_argument('--month', default=3, help='ㅁ month after prognostic ')
    parser.add_argument('--target', default="mortality", help='target column name')
    
    ## Modeling
    parser.add_argument('--seed', default=2024, help='seed')
    parser.add_argument('--kfold', default=5, help='kfold cross validation')
    parser.add_argument('--svm_kernel', default='linear', help='linear, rbf')
    parser.add_argument('--model_name', default='', help='model_name')
    parser.add_argument('--lr_threshold', default=0.5, help='linear regression threshold')
    parser.add_argument('--max_iter', default=100, help='max iteration')
    parser.add_argument('--normalize', default='standard', help='standard, min max, robust')
    parser.add_argument('--balance', default='', help='random_oversampling (ROS), class weighted_random_oversampling (CWROS), SMOTE, SMOTE_Tomek, ADASYN')
    parser.add_argument('--feature_select', default='rfec', help='lda, rfec')
    
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size of each subject for training (default: 16)') 
    parser.add_argument('--valid_batch_size', type=int, default=1, metavar='N', help='valid batch size for training (default: 1)') 
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')
    
    args = parser.parse_args()
    args=vars(args)

    args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code    
    
    if not os.path.isdir(args['save_data']):
        os.makedirs(args['save_data'])
    

    for balance in ['SMOTE', 'ADASYN']:
        print(3, balance)
        args["month"]=3
        args["model_name"]='svm'
        args["normalize"]='robust'
        args["balance"]=balance
        args["n_classes"]=2
        Main.main(args, "ML")

    
    for m in [3]: # , 5, 7, 9, 11
        for model_name in ['xgb']: # 'rf', 'lr', 'svm',
            for balance in ['', 'ROS', 'WROS', 'SMOTE', 'ADASYN']:
                print(m, model_name, balance)
                args["month"]=m
                args["model_name"]=model_name
                args["normalize"]='robust'
                args["balance"]=balance
                args["n_classes"]=2
                Main.main(args, "ML")

    
    # args["month"]=3
    # args["model_name"]='lightGBM'
    # args["normalize"]='robust'
    # args["balance"]='CWROS'
    # args["n_classes"]=5
    # Main.main(args, "ML")
