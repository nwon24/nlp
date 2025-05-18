#!/usr/bin/env python

import numpy as np

n=100000
data=np.ones((n,2))
data[:,0]=np.random.random_sample((n,))*10
data[:,1]=np.sin(data[:,0])
np.savetxt("sin_train.csv",data,fmt="%.6f",delimiter=",",encoding="utf-8")
data[:,0]=np.random.random_sample((n,))*10
data[:,1]=np.sin(data[:,0])
np.savetxt("sin_test.csv",data,fmt="%.6f",delimiter=",",encoding="utf-8")
