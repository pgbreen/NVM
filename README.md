Deep neural network from Newton vs the Machine
===============================================

nvm.py is a python script for generating trajectories, use -h for options 


```sh
$./nvm.py -x -0.25 0.25 -o output
 Initial conditions x2 = (-2.500E-01, 2.500E-01)
 Total time  5.55E-02 sec, per evaluation  1.11E-04 sec
 Saving data to output.dat and plot to output.pdf

```

![Example trajectory](https://github.com/pgbreen/NVM/blob/master/img/output.jpg "Example trajectory")

Preprint available at https://arxiv.org/abs/1910.07291

### Requires 
 - Tensorflow
 - Matplotlib
 - h5py
