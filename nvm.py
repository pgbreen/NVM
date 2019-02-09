#!/usr/bin/env python
import argparse
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#disable AVX FMA support message

parser = argparse.ArgumentParser(description="Generates trajectories using a deep neural network ")

parser.add_argument("-x", help="Coordinate of particle x2, see initial conditions for details ",type=float,default=[-1,-1],metavar=("X21","X22"),nargs=2)

parser.add_argument("-rs", help="random number generator seed (default: None)",
                    type=int,default=None,metavar="rand_seed")

parser.add_argument("-o", help="name of output file (default: \"out\", e.g. out.pdf out.dat)",
                    type=str,default="out",metavar="")


parser.add_argument("-n", help="number of diversions (default: 500)",
                    type=int,default=500,metavar="")

args = parser.parse_args()

random.seed(args.rs)



if args.x[0] == -1 and args.x[1] == -1:
	while args.x[0]**2 + args.x[1]**2 > 1:	
		args.x[0] = random.uniform(-0.5,0.0)
		args.x[1] = random.uniform(0.0,1.0)
	print(f' Random initial conditions x2 = ({args.x[0] : .3E},{args.x[1] : .3E})')
else:
	print(f' Initial conditions x2 = ({args.x[0] : .3E},{args.x[1] : .3E})')
assert (args.x[0] <= 0 and args.x[0] >= -0.5 )," x21 should be in (-0.5,0.0)"
assert (args.x[1] >= 0 and args.x[1] <= 1.0 )," x22 should be in (0.0,1.0)"

model = keras.models.load_model("NN.h5")

lab = np.zeros((args.n,3))
lab[:,0] = np.linspace(0.0,3.9,args.n)
lab[:,1] = args.x[0]
lab[:,2] = args.x[1]

t0 = time.time()
pred = model.predict(lab)
t1 = time.time()

print(f' Total time {t1-t0 : .2E} sec, per evaluation {(t1-t0)/args.n : .2E} sec')

p3 = np.zeros((args.n,2))
p3[:,0] = -pred[:,0]-pred[:,2]
p3[:,1] = -pred[:,1]-pred[:,3]
dat = np.concatenate((pred, p3), axis=1)

np.savetxt(f'{args.o}.dat',dat)

plt.xlim((-1.2,1.2))
plt.ylim((-1.2,1.2))
plt.plot(dat[:,0],dat[:,1],'r-')
plt.plot(dat[0,0],dat[0,1],'ro')
plt.plot(dat[:,2],dat[:,3],'b-')
plt.plot(dat[0,2],dat[0,3],'bo')
plt.plot(dat[:,4],dat[:,5],'g-')
plt.plot(dat[0,4],dat[0,5],'go')
plt.savefig(f'{args.o}.pdf')
plt.clf()
print(f' Saving data to {args.o}.dat and plot to {args.o}.pdf')


