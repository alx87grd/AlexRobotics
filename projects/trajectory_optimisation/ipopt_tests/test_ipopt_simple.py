#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:12:11 2021

@author: alexandregirard
"""
from scipy.optimize import minimize
from cyipopt import minimize_ipopt

def cost (x):
    
    j = x[0]+x[1]
    
    return -j

def constraint_circle(x):
    
    res = x[0]**2+x[1]**2 - 1
    
    return -res


cons = ({'type': 'ineq', 'fun': constraint_circle})
#cons = ({'type': 'eq', 'fun': constraint_circle})

bnds = ((0, None), (0, None))

res = minimize( cost, (0, 0), method='SLSQP', bounds=bnds, constraints=cons)

print(res)

res2 = minimize_ipopt( cost, (0, 0), bounds=bnds, constraints=cons)

print(res2)