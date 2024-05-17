# AI-based-patient-tailored-short-term-stroke-prognosis-prediction-model-using-clinical-information
This is an official repo for 임상정보를 이용한 AI 기반 환자 맞춤형 단기 뇌졸중 예후 예측 모델 (AI-based patient-tailored short-term stroke prognosis prediction model using clinical information)
(대한디지털헬스학회 춘계학술대회, 2024) [\[Paper\]]()

## Description

뇌졸중은 세계적으로 장애 발생의 3대 원인 중 하나이다. 뇌졸중을 진단하고 예측하여 효과적인 재활 및 치료 계획을 수립하는 것이 중요하다. 뇌졸중의 다양성으로 인해 회복 가능성에 큰 차이가 있으며, 실제 임상에서는 각 의료진의 경험과 지식에 따라 환자의 예후를 예측하여 정확도에 편차가 있다. 따라서 효과적인 재활 및 치료 계획 수립을 위하여 인공지능(AI)을 활용한 뇌졸중 환자의 예후 예측 연구가 중요하다. 

본 연구/논문/초록에서는 아래 그림과 같이 병원에 입원한 뇌졸중 환자의 임상 정보를 사용하여 환자의 단기 예후를 예측하는 프레임워크를 제안한다. 그리고 데이터셋의 불균형을 해결하기 위해 다양한 데이터 오버샘플링 기법을 비교하였다.

![](docs/framework.png)

## Getting Started

### Environment Requirement

Clone the repo:

```bash
git clone https://github.com/KDongYoung/AI-based-patient-tailored-short-term-stroke-prognosis-prediction-model-using-clinical-information.git
```

Install the requirements using `conda`:

```terminal
conda create -n env python=3.10.12
conda activate env

```

IF using a Docker, use the recent image file ("pytorch:23.09-py3") uploaded in the [\[NVIDIA pytorch\]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) when running a container


## Data Preparation

First, create a folder `${DATASET_DIR}` to store the data of each subject.

Download the dataset available in the paper "MIMIC-IV, a freely accessible electronic health record dataset" published in *Scientific Data* in 2022.

(Ref: A. E. W. Johnson et al., "MIMIC-IV, a freely accessible electronic health record dataset," Sci. data, Vol. 10, No. 1, pp. 1-9, 2023. doi: [10.1109/TNNLS.2022.3147208](https://doi.org/10.1038/s41597-022-01899-x).)

Unbalanced dataset available in [\[MIMIC-IV\]](https://physionet.org/content/mimiciv/2.2/)

The directory structure should look like this:

```
${DATASET_DIR}
	|--${hosp}
	  |--${admissions.csv.gz}
	  |--${...}
	|--${icu}
    |--${icustays.csv.gz}
	  |--${...}
```

### Training from scratch

```shell script
python TotalMain.py
```

The (pkl file of each model) are saved in `${MODEL_SAVE_DIR}/{seed}/{model_name}` by default

The results are saved in text and csv files in `${MODEL_SAVE_DIR}/{seed}/{Results}/}` by default

The result directory structure would look like this:

```
${MODEL_SAVE_DIR}
 |--${seed}
	  |--${model_name}
	    |--${pkl file}
      |--${txt file}
    |--${Results}
	    |--${csv file}
```

### Evaluation

**The average results (%) for drowsiness classification:**
| Model                      | Accuracy±std. | F1-score±std. | AUROC±std. | 
| -------------------------- | ------------- | ------------- | ---------- |
| LightGBM                   | 96.27±1.10 | 90.85±3.05 | 87.84±4.55 | 
| LightGBM w/ class-weighted random oversampling  |  95.30±0.84  |  89.78±1.96  | 91.12±3.60 | 


## Citation

```
SOON
```

--------------

If you have further questions, please contact dy_kim@chamc.ac.kr

