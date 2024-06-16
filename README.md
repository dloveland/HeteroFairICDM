### Anonymous repo for Unveiling the Impact of Local Homophily on GNN Fairness: In-Depth Analysis and New Benchmarks

In this repo we provide the code to (a) reproduce the data generation outlined in the paper, (b) generate semi-synthetic datasets, and (c) produce the out-of-distributon train-test splits and train/eval models. 

### Data Generation

The base Facebook, Pokec, and Tolokers datasets are not provided in the repo due to size. However, they can be found at:
1. Facebook: https://networkrepository.com/socfb-Penn94.php, https://networkrepository.com/socfb-Cornell5.php
2. Pokec: https://snap.stanford.edu/data/soc-Pokec.html
3. Tolokers: https://github.com/yandex-research/heterophilous-graphs/blob/main/data/tolokers.npz

The processed datasets are provided in the real_datasets folder as *_processed.pt. 

Additionally, the code in 'dataset.py' can be used to generate the processed datasets, or load in the processed datasets. Example code is found in the main function of dataloader. 

### Semi-Synthetic Data Generation

The folder in synthetic datasets is used to generate and house the synthetic data. Specifically, 'generate_synth_data_rewire.py' loads in data and generates a new edge_index for the dataset and saves it. 

Example usage can be found in the bash script 'run_synth.sh'

### Out-of-Distribution Train-Test Splits

Within dataloader.py, we provide the option to stratify the data, i.e. induce the train-test splitting protocol outlined in the paper. This is governed by the gamma parameter. This can either be precomputed, or performed on the fly when training. 

### Training and Evaluation

The code in 'train.py' can be used to train and evaluate the models. The code is set up to train a model on a dataset, either real-world or semi-synth, with the specified train-test protocol. Since the fairness models come from the PyGDebias library, we use a seperate train file, 'train_fair_gnns.py' to train the models. Models can be evaluated using the 'evaluate.py', note that the fair models directly evaluate given the structure of the PyGDebias library and cannot be evaluated through 'evaluate.py'. All model defintions are found in 'models_reg.py', 'models_fairgnn.py', and 'models_nifty.py'. Similar to the semi-synthetic generation, bash scripts are provided to demonstarte how to call train and eval. 