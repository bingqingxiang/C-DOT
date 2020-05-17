# EDGE-FRIENDLY DISTRIBUTED PCA: Experiments
## Table of Contents
<!-- MarkdownTOC -->
- [Synthetic Experiments](#synthetic_experiments)
- [Real-word Experiments](#realworld_experiments)
- [Contributors](#contributors)
<!-- /MarkdownTOC -->


This repo contains the code used for experiments in the [ Edge-Friendly Distributed PCA] thesis.

All of our computation experiments were carried out on a Linux high-performance computing (HPC) cluster provided by Rutgers Office of Advanced Research Computing ( https://oarc.rutgers.edu/amarel/ )

Experiments on small networks with 10 or 20 sites can complete within 10 minutes however some of the larger  network needed about 3 hours.
All of our experiments were done using Python 3.5

In the thesis we conducted two main experiments to produce all plots and charts.

1. Synthetic experiment
2. Real-word experiment


# Synthetic Experiment

## Steps to reproduce
To obtain our results we ran the `Preprocess.ipynb` file which will produce a  folder called `mnist100`. Updload the folder to a cluster and run the 'run.py' file after the code has finished running. Once it is finished copy the generated `.pickle` files to your local machine and run the `plot_synthetic.py` script in python which will produce a plot of the average test error for each algorithm.

## Runtime
On our servers this job completed in 1 hours and 52 minutes. This may vary depending on your computational power.

<a name="contributors"></a>
# Contributors

The original algorithms and experiments were developed by the authors of the thesis
- [Bingqing Xiang]

