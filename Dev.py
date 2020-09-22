import numpy as np
import matplotlib.pyplot as plt

A = np.random.rand(1000,2)
x = np.random.rand(2,1)
temp = np.matmul(A,x)
b = temp + np.random.normal(0,.01,temp.shape)

def f(X, a=A,B=b):
    return 0.5*np.linalg.norm(np.matmul(a,x)-B)**2

def f_grad(X,a=A, B=b):
    return np.matmul(A.transpose(),(np.matmul(a,X)-B))

L = np.max(np.linalg.svd(np.matmul(A,A.transpose()))[1])
lamb = 1/L

