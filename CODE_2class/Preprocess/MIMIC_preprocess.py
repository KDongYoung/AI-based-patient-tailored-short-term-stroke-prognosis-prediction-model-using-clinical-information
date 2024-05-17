import pandas as pd
import numpy as np
from collections import Counter

def main(args, dataset):
    
    ## icd, mimic4(icustays), omr, chartevents, inputevents, outputevents
    # dataset=mimic4
    icd=pd.read_csv(f"{args['save_data']}/icd.csv", low_memory = False)
    omr=pd.read_csv(f"{args['save_data']}/omr.csv", low_memory = False)
    chartevents=pd.read_csv(f"{args['save_data']}/chartevents.csv", low_memory = False)
    inputevents=pd.read_csv(f"{args['save_data']}/inputevents.csv", low_memory = False)
    outputevents=pd.read_csv(f"{args['save_data']}/outputevents.csv", low_memory = False)
    
    icd=find_icd(args['disease_name'], icd)    
    basic=pd.merge(dataset[['subject_id', 'hadm_id', 'gender', 'anchor_age', 'mortality', 'los']], omr, on="subject_id")
    basic['gender'].replace({'M':0, 'F':1}, inplace=True)
    
    if args['over_age']: basic=over_age(basic)
    if args['over_height_over_weight']: basic=over_height_over_weight(basic)
    if args['over_24_icustay']: basic=over_24_icustay(basic)
    # if args['discharge_status']: basic=discharge_status(basic)
    
    # 각 파일 불러오고 merge하고 fill na 하기
    inputevents[np.isinf(inputevents)] = np.nan # inputevents data에 inf 존재
    df=pd.merge(icd[['subject_id', 'hadm_id']], basic, how='left', on = ["subject_id", "hadm_id"])
    icd_mimic_omr_input=pd.merge(df, inputevents, how='left', on = ["subject_id", "hadm_id"])
    icd_mimic_omr_inputoutput=pd.merge(icd_mimic_omr_input, outputevents, how='left', on = ["subject_id", "hadm_id"])
    icd_mimic_omr_inputoutput_chart=pd.merge(icd_mimic_omr_inputoutput, chartevents, how='left', on = ["subject_id", "hadm_id"])
    
    icd_mimic_omr_inputoutput_chart=fill_missing(args, icd_mimic_omr_inputoutput_chart)
    
    print(f"Class별 sample 개수: {Counter(icd_mimic_omr_inputoutput_chart['mortality'])}")
    return icd_mimic_omr_inputoutput_chart.iloc[:, 2:] # subject_id, hadm_id 제외

##################################################### 이 FILLING NAN에 관해서도 추가 다양하게 가능
def fill_missing(args, df):
    # 같은 subject_id이면 mortality가 같을 것이다
    for sidx in df[df["mortality"].isnull()]['subject_id'].unique():
        if df[(df["subject_id"] == sidx) & df["mortality"].notnull()].shape[0]!=0: # mortality 정보가 있는 subject_id
            df.loc[(df["subject_id"] == sidx), "mortality"] = df.loc[(df["subject_id"] == sidx), "mortality"].fillna(df.loc[(df["subject_id"] == sidx) & df["mortality"].notnull(), 'mortality'])
    df=df[df["mortality"].notnull()]
    print(f"Final dataset shape: {df.shape}")
    
    # Fill NaN values in 'column1' with its median, and fill NaN values in other columns with 0
    df.loc[:, args["missing_value"]] = df.loc[:, args["missing_value"]].fillna(np.nanmedian(df.loc[:, args["missing_value"]]))
    
    # 나머지, missing rate가 <30%인 column select
    df=df.loc[:, df.isnull().sum()<df.shape[0]*0.3]
    df = df.fillna(0) # column fillna 0
    return df

def find_icd(disease_name, df):
    print(f"Select {disease_name} related ICD ", end="")
    df=pd.concat([df[df["icd_code"].str.startswith('I63')], df[df["icd_code"].str.startswith('I62')],
                  df[df["icd_code"].str.startswith('I61')], df[df["icd_code"].str.startswith('I60')],
                  df[df["icd_code"].str.startswith('430')], df[df["icd_code"].str.startswith('431')],
                  df[df["icd_code"].str.startswith('432')], df[df["icd_code"].str.startswith('433')],
                  df[df["icd_code"].str.startswith('434')]]) # 13647
        # pd.concat([df[df["icd_code"].str.startswith('I63')], df[df["icd_code"].str.startswith('I62')],
        #           df[df["icd_code"].str.startswith('I61')], df[df["icd_code"].str.startswith('I60')]]) # 공통이 너무 적음(icustay와 concat할 때), 5159
   
    # Replace values matching the pattern with "I63"
    df['icd_code'] = df['icd_code'].str.replace( r'^I60.*', "I60", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^I61.*', "I61", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^I62.*', "I62", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^I63.*', "I63", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^430.*', "430", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^431.*', "431", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^432.*', "432", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^433.*', "433", regex=True)
    df['icd_code'] = df['icd_code'].str.replace( r'^434.*', "434", regex=True)
    
    df.reset_index(drop=True, inplace=True)
    print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df
    
def over_age(df):
    age=18
    print("Include subject over 18 ", end="")
    df=df.astype({'anchor_age':int})
    df=df[df["anchor_age"]>=age]
    print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def over_height_over_weight(df):
    print(f"Include subject height (50~250cm) and weight (50~300kg)", end=" ")
    # ## 단위를 고려하면 743 row로 줄어듦 -> 90 subject   HEIGHT 2.54, WEIGHT 2.2
    df=df[(50.0 <= df["height"]) & (df["height"]<= 250.0) & 
          (50.0 <= df["weight"]) & (df["weight"]<= 300.0)]
    print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df

def over_24_icustay(df):
    print("Include subject icu 24hr stay ", end="")
    df=df[df["los"]>=1] # los=1은 1day
    print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
    return df
    
# def discharge_status(df):
#     print("Exclude discharge status missing subject: ", end="")
#     df=df[df['dischtime'].notnull()]
#     print(df[['subject_id', 'hadm_id']].drop_duplicates().shape[0])
#     return df