# EDGE-FRIENDLY DISTRIBUTED PCA: Experiments
## Table of Contents

<!-- MarkdownTOC -->
- [General Information](#general_information)
- [Synthetic Experiments](#synthetic_experiments)
- [Real-word Experiments](#realworld_experiments)
- [Contributors](#contributors)
<!-- /MarkdownTOC -->


# General Information
This repo contains the code used for experiments in  the [ Edge-Friendly Distributed PCA] thesis and [Distributed Principal Subspace Analysis for
Partitioned Big Data: Algorithms, Analysis, and Implementation] based on a high-performance computing cluster with Message Passing Interface (MPI).

All of our computation experiments were carried out on a Linux high-performance computing (HPC) cluster provided by Rutgers Office of Advanced Research Computing (https://oarc.rutgers.edu/amarel/)

Experiments on small networks with 10 or 20 sites can complete within 10 minutes however some of the larger networks needed about 3 hours. 

The dependencies and versions for our codes are as follows.
1. Python/3.5.2
2. intel/18.0.5
3. openmpi/2.1.1

In this paper, we conducted two main experiments to produce all plots and charts.
1. Synthetic experiment
2. Real-world experiment(not include imagenet experiment)
3. Imagenet experiment

The DistributedOI folder includes all classes and functions needed for our MPI experiments.
1. Algorithms.py: Experiment() class for non-imagenet experiments
2. Algorithms_ImageNet.py: Experiment() class for imagenet experiments
3. data_prep.py: functions used for calculating the true eigengap, load synthetic datasets, and real-world datasets, and split Column-wise distribute data into (number_of_nodes) files.
4. GraphTopology.py: generated graph topology and get the corresponding weight matrix.

The getImagenet folder provided code to preprocess imagenet dataset.
1. run.sh: run this file on amarel cluster will process all .png images in a folder containing imagenet images and give us the data folder for imagenet data with several pickle files where each pickle contains 5000 images.
2. resizeImg.py: the python code runs within run.sh
3. svd_image32.py : calculate svd for imagenet data, need to define the number of nodes and r(top r eigenvector)

The Preprocess.ipynb file is mainly used in the data preprocessing part on our local machine. This file will give us the data folder with distributed data as pickle files with defined experiment parameters. In addition, this file will calculate top r low-rank subspace, graphs and the corresponding weight matrix and save them as pickle files.

The experiment.sh file is mainly used to run experiments on the ameral cluster in a distributed fashion for non-imagenet experiments which correspond to run.py file.

The imagenet.sh file is mainly used to run experiments on ameral cluster in a distributed fashion for imagenet experiments which correspond to run_imagenet.py file.

# Synthetic Experiment

## Steps to reproduce
To obtain our results we uncomment the Design Data block and ran the `Preprocess.ipynb` file which will produce a  folder called `synthetic_n{}r{}g{}` which includes n(number of nodes) pickle files. Updload the folder to a cluster and run the 'run.py' file through experiment.sh file with the experiment parameters we had. Once it is finished copy the generated `.pickle` files to your local machine and run the `plot_synthetic.py` script in python which will produce a plot of the average test error for each algorithm.

## Runtime
On our servers, this job was completed in 30 minutes. This may vary depending on your computational power.

# Realworld Experiments
The first step is to download the real-world datasets from official websites. To obtain our results we uncomment the corresponding block and ran the `Preprocess.ipynb` file which will produce a  folder called `{dataset's name}{number of nodes}` which include n(number of nodes) pickle files. Upload the folder to a cluster and run the 'run.py' file through experiment.sh file with the experiment parameters we had. Once it is finished copy the generated `.pickle` files to your local machine and run the `plot_synthetic.py` script in python which will produce a plot of the average test error for each algorithm.

## Runtime
On our servers, this job was completed in 1 hour and 52 minutes. This may vary depending on your computational power.


<a name="contributors"></a>
# Contributors

The original algorithms and experiments were developed by the authors of the thesis
- [Bingqing Xiang]

