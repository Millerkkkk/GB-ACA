# GB-ACA Experiment Code

This repository contains the experiment code and instances for the GB-ACA (Granular Balls - Ant Colony Algorithm) as implemented in the `GB-ACA_final.py` script.

## Contents

- `A_First_Stage.py`: Stage 1: Division of the entire dataset into clusters according to the loading capacity of the vehicles.
- `GB-ACA_final.py`: Stage 2: Route planning. The main experiment code implementing the GB-ACA algorithm.
- `large_instances(100customer21cs_10)`: The dataset required for the experiment.
- `Problem.pdf`: Contains the problems encountered in the experiment and some of the results of the experiment.

## The Core Idea of the GB-ACA Experiment

1. **Step 1**: Split each cluster into multiple granular-balls using K-means clustering.
2. **Step 2**: Consider each granular-ball as a point (represented by the center of the granular-ball in the experiment) and plan the paths of the granular-balls using the ACA algorithm to obtain a path among the granular-balls.
3. **Step 3**: Following the order of the granular-ball paths obtained in Step 2, use ACA to plan the internal paths within each granular-ball in turn.
