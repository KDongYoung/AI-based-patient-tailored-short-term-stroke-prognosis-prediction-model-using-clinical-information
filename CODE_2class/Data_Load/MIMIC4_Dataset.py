import pandas as pd
import os
import gzip
import gc
from datetime import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

class MIMIC4_Dataset():
    def __init__(self, args) -> None:
        self.args=args
        self.hosp_path = args['data_root']+"hosp/"
        self.icu_path = args['data_root']+"icu/"        
        self.disease_name=args['disease_name']
        self.save_path=args['save_data']

        self.dataset={}
        if not os.path.isdir(f"{args['save_data']}"):
            os.makedirs(args['save_data'])
        if "mimic4.csv" not in os.listdir(args['save_data']):
            d_items_csv = gzip.open(f'{self.icu_path}/d_items.csv.gz')
            self.d_items = pd.read_csv(d_items_csv)
            # d_labitems_csv = gzip.open(f'{self.hosp_path}/d_labitems.csv.gz')
            # self.d_labitems = pd.read_csv(d_labitems_csv)
        
            self.basic() 
            self.omr_height_weight_bloodpressure() 
            self.diagnoses_icd() 
            
            self.inputevents()  
            self.outputevents()  
            self.chartevents()
            
                        
        args["name"]=[n.split(".")[0] for n in os.listdir(self.save_path)]
        
        with open(self.args['save_data'] + '/args.txt', 'a') as f:
            f.write(datetime.now().strftime('%Y%m%d'))
            
    def basic(self):
        print(f"Preprocess icustays.csv, admissions.csv, patients.csv", end=" ")
        start=datetime.now()
        
        ## read 'icustays', 'admissions', 'patients' files
        icu_stay_csv = gzip.open(f'{self.icu_path}/icustays.csv.gz')
        icu_stay = pd.read_csv(icu_stay_csv)
        admission_csv = gzip.open(f'{self.hosp_path}/admissions.csv.gz')
        admission = pd.read_csv(admission_csv)
        patients_csv = gzip.open(f'{self.hosp_path}/patients.csv.gz')
        patients = pd.read_csv(patients_csv)

        # remove unneeded columns
        icu_stay = icu_stay[['subject_id', 'hadm_id', 'intime', 'outtime', 'los']]
        admission = admission[['subject_id', 'hadm_id', 'hospital_expire_flag', 'admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime']]
        patients = patients[['subject_id', 'gender', 'anchor_age', 'dod']]

        icu_stay = pd.merge(icu_stay, patients, on='subject_id', how='left')
        icu_stay = pd.merge(icu_stay, admission, on=['subject_id','hadm_id'], how='left')
        
        icu_stay['dod'].fillna("2210-12-31", inplace=True) ## NaT는 데이터를 작성할 때까지 생존했음을 의미하는 것으로 고려
        cols = ['intime', 'outtime', 'admittime', 'dischtime', 'deathtime', 'edregtime', 'edouttime', 'dod']
        icu_stay[cols] = icu_stay[cols].apply(pd.to_datetime)
        
        icu_stay.loc[icu_stay['hospital_expire_flag']==0, 'dischtime']
        # dod only has date (no time)
        icu_stay.loc[icu_stay['deathtime'].notnull(), 'dischtime']=icu_stay.loc[icu_stay['deathtime'].notnull(), 'deathtime'] # deathtime이 있는 경우
        icu_stay.loc[icu_stay['deathtime'].notnull(), 'dod']=icu_stay.loc[icu_stay['deathtime'].notnull(), 'deathtime'] # deathtime이 있는 경우
        icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==1), 'dod']=icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==1), 'dischtime'] # deathtime은 없는데 hospital in-mortality
        icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==0), 'dod']=icu_stay.loc[(icu_stay['deathtime'].isnull()) & (icu_stay['hospital_expire_flag']==0), 'dod'].apply(lambda x: x + pd.Timedelta(hours=23, minutes=59, seconds=59)) # deathtime 없는 경우
        
        icu_stay['mortality'] = (icu_stay['dod'] - icu_stay['dischtime']).dt.total_seconds()

        # mortality가 양수 혹은 null(사망하지 않음)인 데이터만 사용
        self.icu_stay = icu_stay[(icu_stay.mortality >= 0) | (icu_stay.mortality.isnull())] # dod < dischtime 제거
        
        if len(self.args["missing_value"])==0: self.args["missing_value"].extend(['gender','anchor_age', 'mortality', 'los'])
        self.icu_stay.to_csv(f"{self.save_path}/mimic4.csv", index=False)
        print(f'{datetime.now()-start} (hh:mm:ss.ms)')
                
    def outputevents(self):
        print(f"Preprocess outputevents.csv", end=" ")
        start=datetime.now()
        
        outputevents_csv = gzip.open(f'{self.icu_path}/outputevents.csv.gz')
        outputevents = pd.read_csv(outputevents_csv)

        del outputevents_csv 
        gc.collect()

        outputevents = pd.merge(outputevents[['subject_id', 'hadm_id', 'storetime','itemid', 'value']], 
                                self.icu_stay[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='left')

        outputevents['intime'] = pd.to_datetime(outputevents['intime'])
        outputevents['storetime'] = pd.to_datetime(outputevents['storetime'])
        outputevents['dur_store'] = (outputevents['storetime'] - outputevents['intime']).dt.total_seconds()

        # 입원 이후의 데이터
        outputevents = outputevents[outputevents['dur_store'] > 0]
        outputevents = pd.merge(outputevents, self.d_items[['itemid', 'label']], on=['itemid'], how='left')

        # 입원 이후의 value의 평균을 사용
        tmp = outputevents.groupby(['subject_id','hadm_id', 'label'])['value'].mean()
        tmp = pd.DataFrame(tmp).reset_index()
        outputevents_pivot = tmp.pivot(index=['subject_id','hadm_id'], columns='label', values='value').reset_index()
        
        outputevents_pivot.to_csv(f"{self.save_path}/outputevents.csv", index=False)
        print(f'{datetime.now()-start} (hh:mm:ss.ms)')
        
    def inputevents(self):
        print(f"Preprocess inputevents.csv", end=" ")
        start=datetime.now()
        
        inputevents_csv = gzip.open(f'{self.icu_path}/inputevents.csv.gz')
        inputevents = pd.read_csv(inputevents_csv)

        del inputevents_csv 
        gc.collect()

        inputevents = pd.merge(inputevents[['subject_id', 'hadm_id', 'starttime', 'endtime', 'storetime', 'itemid', 'amount']], 
                                self.icu_stay[['subject_id', 'hadm_id', 'intime']], on=['subject_id', 'hadm_id'], how='left')
        
        inputevents['intime'] = pd.to_datetime(inputevents['intime'])
        inputevents['storetime'] = pd.to_datetime(inputevents['storetime'])
        inputevents['endtime'] = pd.to_datetime(inputevents['endtime'])
        inputevents['starttime'] = pd.to_datetime(inputevents['starttime'])
        inputevents['dur_input'] = (inputevents['endtime'] - inputevents['starttime']).dt.total_seconds()
        inputevents['valueRate']=inputevents['amount']/(inputevents['dur_input']) # 단위/초
        inputevents['dur_store'] = (inputevents['storetime'] - inputevents['intime']).dt.total_seconds()
        inputevents = pd.merge(inputevents, self.d_items[['itemid', 'label']], on=['itemid'], how='left')
        inputevents = inputevents[inputevents['dur_store'] > 0]

        # value의 평균을 사용
        inputevents = pd.DataFrame(inputevents.groupby(['subject_id', 'hadm_id', 'label'])['valueRate'].mean()).reset_index()
        inputevents_pivot = inputevents.pivot(index=['subject_id','hadm_id'], columns='label', values='valueRate').reset_index()
        
        inputevents_pivot.to_csv(f"{self.save_path}/inputevents.csv", index=False)
        print(f'{datetime.now()-start} (hh:mm:ss.ms)')
        
    def chartevents(self):
        print(f"Preprocess chartevents.csv")
        start=datetime.now()

        chartevent_csv = gzip.open(f'{self.icu_path}/chartevents.csv.gz')

        chartevent_result = pd.DataFrame()
        cols = ['subject_id', 'hadm_id', 'storetime', 'itemid', 'valuenum']

        # chartevent 용량이 커서 부분부분 불러서 합치기
        for cnt, df in enumerate(pd.read_csv(chartevent_csv, chunksize=1e7, usecols=cols)):
            print(f"{cnt+1} chunk is added in chartevent.csv")
            df = pd.merge(df, self.icu_stay[['subject_id','hadm_id', 'intime']], on=['subject_id','hadm_id'], how='left')
            df['intime'] = pd.to_datetime(df['intime'])
            df['storetime'] = pd.to_datetime(df['storetime'])
            df['dur_store'] = (df['storetime'] - df['intime']).dt.total_seconds()

            # # 6시간 이내의 데이터
            df = df[df['dur_store'] > 0]
            df = pd.merge(df, self.d_items[['itemid', 'label']], on=['itemid'], how='left')
            chartevent_result = pd.concat([chartevent_result, df])
            
        chartevent_result = pd.DataFrame(chartevent_result.groupby(['subject_id', 'hadm_id', 'label'])['valuenum'].mean()).reset_index()
        chartevent_result = chartevent_result.pivot(index=['subject_id','hadm_id'], columns='label', values='valuenum').reset_index()
        
        chartevent_result.to_csv(f"{self.save_path}/chartevents.csv", index=False)
        print(f'{datetime.now()-start} (hh:mm:ss.ms)')

    def omr_height_weight_bloodpressure(self):
        print(f"Change the height, weight, blood pressure value as column", end=" ")
        start=datetime.now()
        
        csv = gzip.open(f"{self.hosp_path}/omr.csv.gz")
        omr=pd.read_csv(csv, low_memory = False)

        omr['result_name'].replace('Height (Inches)', 'Height', inplace=True)
        omr['result_name'].replace('Weight (Lbs)', 'Weight', inplace=True)

        # Filter the DataFrame to include only "Height" and "Weight" and "Blood Pressure" entries
        height_df = pd.DataFrame(omr[omr['result_name'] == 'Height'], columns=['subject_id','chartdate','result_value'])
        height_df.columns=['subject_id','chartdate', 'height']
        weight_df = pd.DataFrame(omr[omr['result_name'] == 'Weight'], columns=['subject_id','chartdate','result_value'])
        weight_df.columns=['subject_id','chartdate', 'weight']
        bp_df = pd.DataFrame(omr[omr['result_name'] == 'Blood Pressure'], columns=['subject_id','chartdate','result_value'])
        bp_df["blood_pressure_max"]=[row.split("/")[0] for row in bp_df["result_value"]]
        bp_df["blood_pressure_min"]=[row.split("/")[1] for row in bp_df["result_value"]]
        bp_df=bp_df[['subject_id', 'chartdate', 'blood_pressure_max', 'blood_pressure_min']]

        df=pd.merge(height_df, weight_df, on=['subject_id', 'chartdate'])
        df=pd.merge(df, bp_df, on=['subject_id', 'chartdate'])
        df=df.astype({'height':float, 'weight':float, 'blood_pressure_max':float, 'blood_pressure_min':float})
        df=df.groupby(['subject_id'])['height', 'weight', "blood_pressure_max", "blood_pressure_min"].mean()
        df=pd.DataFrame(df).reset_index()
        
        if len(self.args["missing_value"])<=2: self.args["missing_value"].extend(["height", "weight", "blood_pressure_max", "blood_pressure_min"])
        df.to_csv(f"{self.save_path}/omr.csv", index=False) 
        print(f'{datetime.now()-start} (hh:mm:ss.ms)')    
    
    def diagnoses_icd(self):
        print(f"Merge dataset for diagnoses icd", end=" ")
        start=datetime.now()
        
        csv = gzip.open(f"{self.hosp_path}/diagnoses_icd.csv.gz")
        icd=pd.read_csv(csv, low_memory = False)
        csv = gzip.open(f"{self.hosp_path}/d_icd_diagnoses.csv.gz")
        d_icd=pd.read_csv(csv, low_memory = False)

        # icd=icd[icd['seq_num']<=5]
        df=pd.merge(icd[['subject_id','hadm_id','icd_code']], d_icd[['icd_code', "long_title"]], on='icd_code', how='left')
        df.to_csv(f"{self.save_path}/icd.csv", index=False) 
        print(f'{datetime.now()-start} (hh:mm:ss.ms)')   
        
    def __getitem__(self):
        pass
        # X = self.X[idx].astype('float32')  # for only eeg
        # y = self.y[idx].astype('int64') 
        
        # X=np.expand_dims(X,axis=0) # (1, channel, time) batch shape
    
        # return X, y, self.subj_id