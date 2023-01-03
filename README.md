# Certifying Some Distributional Fairness with Subpopulation Decomposition

The repository contains the official implementation of [Certifying Some Distributional Fairness with Subpopulation Decomposition](https://arxiv.org/abs/2205.15494) [NeurIPS 2022].

## Download and Installation
The required packages can be installed by

```pip install -r requirement.txt```

## Usage
### Pretrain the models
Train the models to be certified. We perform vanilla training on six commonly used datasets for fairness evaluation.

```python train.py -dataset DATASET -num_epochs 100 -batch-size 256 -learning_rate 0.001```

DATASET can be chosen as one in {'adult','compas','crime','german','health','lawschool'}.
We use the learning rate 0.05 for crime and german dataset and 0.001 for other datasets. 

### Pretrain the models on Gaussian mixture data
Train the models to be certified on Gaussian mixture data.

```python train_gaussian_mixture.py```

### Fairness certification for sensitive shifting

Provide fairness certification of pretrained models (or any other models) for sensitive shifting (no distribution shifting within each subgroup).

```python certify_fairness.py -num_generated_distributions 1000 -certificate_low 0.1 -certificate_high 0.5 -num_points_bound 10 -use_loss 0```

``num_generated_distributions`` denotes the number of generated simulation distribution and the corresponding evaluation results. 
``[certificate_low, certificate_high]`` is the interval of distance for certification and ``num_points_bound`` is the number of certification points in the interval.
We can set ``use_loss`` 0 for certification of classification error and 1 for certification of the specified loss.

### Fairness certification for general shifting

Provide fairness certification of pretrained models (or any other models) for general shifting (the distribution of group proportion and the distribution within subgroups can both shift).

```python covariate_shifting_train_and_test.py -num_generated_distributions 1000 -certificate_low 0.1 -certificate_high 0.5 -num_points_bound 10 -use_loss 0 -interval_kr 0.005```

``interval_kr`` is the width of grids used to divide the optimization problem.

## Citation
If this code is helpful for your study, please cite:
```
@inproceedings{
kang2022certifying,
title={Certifying Some Distributional Fairness with Subpopulation Decomposition},
author={Mintong Kang and Linyi Li and Maurice Weber and Yang Liu and Ce Zhang and Bo Li},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=6mej19W1ppP}
}
```
## Contact
If you have any questions, you can contact ``mintong2@illinois.edu``.