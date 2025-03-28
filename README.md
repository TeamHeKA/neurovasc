# Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs
This repository is the official PyTorch implementation of the experiments for clinical outcome prediction task, as described in the following paper:

Jhee, J. H., Megina, A., Beaufils, P. C. D., Karakachoff, M., Redon, R., Gaignard, A., Coulet, A. [Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs](https://arxiv.org/abs/2502.21138). (ESWC 2025)

## Installation & Dependencies
The code is mainly tested on Python 3.11 and a Linux OS.

Dependencies:
* numpy 1.26.4
* pandas 2.2.2
* torch 2.4.1
* torch-geometric 2.6.1
* scikit-learn 1.5.1
* rdflib 7.0.0
* joblib 1.4.2

## Run the demo
The "notebooks" folder contains demos for each step, from generating synthetic data to predicting patient outcomes.

## Reproducibility
To reproduce the experiments in the paper: 
```
cd exp
python3 main_sphn_rgcn.py
```
The default option reproduces results on SPHN-trs with RGCN+lit. For other experiments check "-h" help function.
```
python3 main_sphn_rgcn.py -h
```
The results will be saved in the "exp/result" folder. 

## Citation
```
@article{jhee2025predicting,
  title={Predicting clinical outcomes from patient care pathways represented with temporal knowledge graphs},
  author={Jhee, Jong Ho and Megina, Alberto and Beaufils, Pac{\^o}me Constant Dit and Karakachoff, Matilde and Redon, Richard and Gaignard, Alban and Coulet, Adrien},
  journal={arXiv preprint arXiv:2502.21138},
  year={2025}
}
```
