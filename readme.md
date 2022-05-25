# EMP_dynamics

The demo code for paper _Revisiting Neural Network Dynamics via Estimation-Measure-Oriented Bottlenecks_. 

## Run

1. Train the main network for MNIST calssification task. 
   1. Run src.pretrain.py file.
2. Use measure estimators to do the measurements and obtain the results at each epoch.
   1. Fill the path of training results of src.pretrain.py into the sim.train_measure.py file.
   2. Run sim.train_measure.py file.

## Python environment dependency
+ Python >= 3.9.7
+ pytorch >= 1.11.0
+ numpy >= 1.21.5
+ tqdm >= 4.64.0
