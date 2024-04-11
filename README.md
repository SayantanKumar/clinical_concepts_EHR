# Concept-based explanations 
This repository contains official implementation for our paper titled "Self-explaining Neural Network with Concept-based
Explanations for ICU Mortality Prediction" [Paper](https://dl.acm.org/doi/pdf/10.1145/3535508.3545547) and our NeurIPS 2023 Workshop paper "Explaining Longitudinal Clinical Outcomes using Domain-Knowledge driven Intermediate Concepts" [Paper](https://openreview.net/forum?id=hpuOA3nkVW)

![Workflow](figures/workflow.png)


**Abstract.** The black-box nature of complex deep learning models makes it challenging to explain the rationale behind model predictions to clinicians and healthcare providers. Most of the current explanation methods in healthcare provide expla- nations through feature importance scores, which identify clinical features that are important for prediction. For high-dimensional clinical data, using individual input features as units of explanations often leads to noisy explanations that are sensitive to input perturbations and less informative for clinical interpreta- tion. In this work, we design a novel deep learning framework that predicts domain-knowledge driven intermediate high-level clinical concepts from input features and uses them as units of explanation. Our framework is self-explaining; relevance scores are generated for each concept to predict and explain in an end-to-end joint training scheme. We perform systematic experiments on a real-world electronic health records dataset to evaluate both the performance and explainability of the predicted clinical concepts.


## Environment
- We recommend an evironment with python >= 3.7 and pytorch >= 1.10.2, and then install the following dependencies:
```
pip install -r requirements.txt
```

## Data extraction from MIMIC IV and preprocessing
The **MIMIC_data_extaction.py** script is used to extract the time-series and the time-invariant features from the MIMIC-IV v0.4 dataset. Instructions for downloading data from the MIMIC IV v0.4 dataset can be found [here](https://physionet.org/content/mimiciv/0.4/).  This script also calculates the ground truth labels for concepts (SOFA otrgan-falure-risk scores) and the clinical outcome (ICU mortality). 

For each patient, we extracted 87 time-series features and 24 static features which included laboratory test results, vital signs, comorbidities, admission information and demographics. The MIMIC files used in our study are as follows: 
```
- patients.csv
- admissions.csv
- chartevents.csv
- transfers.csv
- diagnoses_icd.csv
- d_icd_diagnoses.csv
- d_items.csv
- labevents.csv
- inputevents.csv
- outputevents.csv
```
The functions for extracting relevant features from each of the MIMIC files are given in the script **utils_data_extraction.py**. 


## Feature preprocessing
The **data_splitter.py** script is used for splitting the data into train-validation-test (75:15:10), scaling variables and converting dataframes into tensors for input to the model. Feature pre-processing of time-series variables include clipping the outlier values to the 1st and 99th percentile values and standardization using the RobustScalar package from sklearn. 

## Model training and evaluation

All components of our proposed architecture have been implemented in the **model.py** script, including the recurrent module with time-series module, concept and relevance network. The **train_evaluate.py** script has the complete training module and the evaluation functions for generating model performance (AUROC/AUPRC) and model explanations. 

![alt-text-1](figures/performance_table.png){: style="height:83px"}
![alt-text-2](figures/SOFA_explanation.png){: style="height:83px"}

## Citation
If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@inproceedings{kumar2022self,
  title={Self-explaining neural network with concept-based explanations for ICU mortality prediction},
  author={Kumar, Sayantan and Yu, Sean C and Kannampallil, Thomas and Abrams, Zachary and Michelson, Andrew and Payne, Philip RO},
  booktitle={Proceedings of the 13th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},
  pages={1--9},
  year={2022}
}

@inproceedings{kumar2023explaining,
  title={Explaining Longitudinal Clinical Outcomes using Domain-Knowledge driven Intermediate Concepts},
  author={Kumar, Sayantan and Kannampallil, Thomas and Sotiras, Aristeidis and Payne, Philip},
  booktitle={XAI in Action: Past, Present, and Future Applications},
  year={2023}
}
```
