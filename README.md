# Learning to Predict with Supporting Evidence: Applications to Clinical Risk Prediction

## Description
This repository provides code to accompany the paper: Learning to Predict with Supporting Evidence: Applications to Clinical Risk Prediction, at ACM CHIL 2021.

The file `vem.py` implements the variational EM procedure described in the paper, and the file `inf.py` implements the MAP inference model.  Due to data use agreements, the clinical dataset used in the original work cannot be shared here. Skeleton code for data loading is provided in `utils.py`. 

To run variational EM for model learning after specifying dataset details, run: 

`python vem.py SEED`

replacing `SEED` with the random seed of choice.


To run MAP inference to learn the final predict/supporting evidence model, after specifying dataset details and running vEM, run: 

`python inf.py SEED`

replacing `SEED` with the random seed of choice.


