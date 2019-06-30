
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:28:41 2018

@author: Elias
"""
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import norm
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def sphere_radius_estimate(p,n,c):
	R = []
	for i in range(0,len(p)):
		def objective_fun(t):
			x = np.array([p[i][0]+(p[i][0]-cam[0])*t, p[i][1]+(p[i][1]-cam[1])*t, p[i][2]+(p[i][2]-cam[2])*t])
			tmp = np.array([x[0]-c[0], x[1]-c[1], x[2]-c[2]])
			res = norm(np.cross(tmp.T, c.T))
			return res
		t_lqm = least_squares(objective_fun, 0)
		x_lqm = np.array([p[i][0]+(p[i][0]-cam[0])*t_lqm.x, p[i][1]+(p[i][1]-cam[1])*t_lqm.x, p[i][2]+(p[i][2]-cam[2])*t_lqm.x])
		tmp = np.array([x_lqm[0]-c[0], x_lqm[1]-c[1], x_lqm[2]-c[2]])
		R.append(norm(tmp))
	return np.mean(R)

def sphere_centre_estimate(p, n):
    sum1 = np.array([[0,0],[0,0]])
    sum2 = np.array([[0],[0]])
    for i in range(0, len(p)):
        ni = np.reshape(n[i][0:2], (2,1))
        pi = np.reshape(p[i][0:2], (2,1))
        
        sum1 = sum1 + np.eye(2,2) - np.dot(ni,np.transpose(ni))
        sum2 = sum2 + np.dot(np.eye(2,2) - np.dot(ni,np.transpose(ni)), pi)
  
    res = np.dot(np.linalg.pinv(sum1),sum2)
    return res
		

cam = np.array([0,0,0])
p =[ np.array([15,2,8]), np.array( [35,6,0])] 
n =[ np.array([1,7,0]), np.array( [29,7,0])]
  

cam = np.array([-1,2,5])
ctmp = sphere_centre_estimate(p,n)
c = np.array([ctmp[0], ctmp[1], [0]])
print('centre')
print(c)
#r =  sphere_radius_estimate(p,n, c)
#print(r)


 