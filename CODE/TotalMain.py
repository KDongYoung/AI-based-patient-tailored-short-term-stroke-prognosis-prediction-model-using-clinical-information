"""
Paper "AI-based patient-tailored short-term stroke prognosis prediction model using clinical information" implementation code (official)
"임상정보를 이용한 AI 기반 환자 맞춤형 단기 뇌졸중 예후 예측 모델" 논문 구현 코드
Author: Dong-Young Kim
Date: May 17, 2024
Email: dy_kim@chamc.co.kr
Organization: MIH Lab, CHA University / CHA Future 
"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

import Main

import argparse
import os

if __name__ == '__main__':
    """ Experiment Setting """ 
    # ARGUMENT
    parser = argparse.ArgumentParser(description='Stroke analysis')
    parser.add_argument('--data_root', default='data/DATASET/mimic4_2.2/', help="name of the data folder")
    parser.add_argument('--run_code_folder', default='code path')
    parser.add_argument('--save_data', default='data/pre_MIMIC/', help="name of the data folder") 
    parser.add_argument('--save_root', default='data/KSDH_MIMIC4/', help="where to save the models") 
    parser.add_argument('--result_dir', default="", help="save folder name") 
    parser.add_argument('--total_path', default="", help='total result path')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda')
    parser.add_argument('--cuda_num', type=int, default=0, help='cuda number')
    parser.add_argument('--device', default="", help='device')
    parser.add_argument('--dataset_name', default='mimic4', help='dataset name: mimic4')
    parser.add_argument('--internal', default='mimic4', help='dataset name: mimic4')
    parser.add_argument('--disease_name', default='Stroke')
    
    parser.add_argument('--n_classes', type=int, default=0, help='num classes')
    
    ## MIMIC-IV 
    parser.add_argument('--target', default='mortality', help='target feature')
    parser.add_argument('--name', default=[], help='name of files used for analysis')
    parser.add_argument('--over_age', default=True, help='preprocess over age 18')
    parser.add_argument('--over_24_icustay', default=True, help='preprocess subject that have stayed in the icu for more than 24 hours')
    parser.add_argument('--over_height_over_weight', default=True, help='preprocess subject over height and weight')
    parser.add_argument('--missing_value', default=['gender', 'anchor_age', 'los', 'height', 'weight', 'blood_pressure_max', 'blood_pressure_min'], 
                        help='columns that needs to be considered while fill na, not just fillna(0)')
    parser.add_argument('--month', default=3, help='ㅁ month after prognostic')
    
    ## Modeling
    parser.add_argument('--seed', default=2024, help='seed')
    parser.add_argument('--kfold', default=5, help='how much fold cross validation')
    parser.add_argument('--svm_kernel', default='linear', help='kernel of svm: linear, rbf')
    parser.add_argument('--model_name', default='', help='model_name')
    parser.add_argument('--lr_threshold', default=0.5, help='linear regression threshold')
    parser.add_argument('--max_iter', default=1000, help='max iteration')
    parser.add_argument('--normalize', default='', help='standard, min max, robust')
    parser.add_argument('--balance', default='', help='random_oversampling (ROS), weighted_random_oversampling (WROS), SMOTE, SMOTE_Tomek, ADASYN')
    parser.add_argument('--feature_select', default='rfec', help='rfec')
    
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size of each subject for training (default: 16)') 
    parser.add_argument('--valid_batch_size', type=int, default=1, metavar='N', help='valid batch size for training (default: 1)') 
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 1)')

    args = parser.parse_args()
    args=vars(args)

    args["run_code_folder"]=os.path.realpath(__file__) # folder name of running code    
    
    if not os.path.isdir(args['save_data']):
        os.makedirs(args['save_data'])
    
    
    args["month"]=3
    args["normalize"]='robust'
    args["n_classes"]=2
    
    for model_name in ['rf', 'lr', 'svm', 'xgb', 'gbt', 'knn']:
        print(model_name)
        args["model_name"]=model_name
        args["balance"]=''
        Main.main(args, "ML")
    
    args["model_name"]='lightGBM'
    for balance in ['', 'ROS', 'WROS', 'SMOTE', 'ADASYN']:
        print(model_name, balance)
        args["balance"]=balance
        Main.main(args, "ML")
