# The Demos

## 1. Synthetic data generation
* Run the "Data Generation/synthetic_data_generation.ipynb" file to generate synthetic tabular data.
* Synthetic data is saved in "data/syn_data.csv" folder.

## 2. Graph Generation
* Run "Data Generation/SPHN-graph_generation.ipynb" for SPHN graph generation.
* Run "Data Generation/CARESM-graph_generation.ipynb" for SPHN graph generation.
* Graphs are saved in "data/xxx.nt" folder.

## 3-1. Running RF on tabular data
* Run "Tabular/baseline_rf.ipynb" to reproduce the result of random forest model.

## 3-2. Running RGCN on graph data
* Run "Graphs/data_preprocess_xxx.ipynb" for preprocessing the graph data.
* Run "Graphs/rgcn_node_pred.ipynb" to reproduce the result of RGCN+lit model.
* The results are saved in "Graphs/result" folder.