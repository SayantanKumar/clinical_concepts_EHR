# concept_based_explanations
This repository contains official implementation for our paper titled "Self-explaining Neural Network with Concept-based
Explanations for ICU Mortality Prediction" [Paper](https://dl.acm.org/doi/pdf/10.1145/3535508.3545547) and our NeurIPS 2023 Workshop paper "Explaining Longitudinal Clinical Outcomes using Domain-Knowledge driven Intermediate Concepts" [Paper](https://openreview.net/forum?id=hpuOA3nkVW)

![Workflow](figures/workflow.png)

![alt-text-1](figures/performance_table.png) ![alt-text-2](figures/SOFA_explanation.png)

**Abstract.** The black-box nature of complex deep learning models makes it challenging to explain the rationale behind model predictions to clinicians and healthcare providers. Most of the current explanation methods in healthcare provide expla- nations through feature importance scores, which identify clinical features that are important for prediction. For high-dimensional clinical data, using individual input features as units of explanations often leads to noisy explanations that are sensitive to input perturbations and less informative for clinical interpreta- tion. In this work, we design a novel deep learning framework that predicts domain-knowledge driven intermediate high-level clinical concepts from input features and uses them as units of explanation. Our framework is self-explaining; relevance scores are generated for each concept to predict and explain in an end-to-end joint training scheme. We perform systematic experiments on a real-world electronic health records dataset to evaluate both the performance and explainability of the predicted clinical concepts.


## Environment
- We recommend an evironment with python >= 3.7 and pytorch >= 1.10.2, and then install the following dependencies:
```
pip install -r requirements.txt
```

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
