#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 11:12:11 2021

@author: alexandregirard
"""
from scipy.optimize import minimize
from cyipopt import minimize_ipopt

fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2

cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6}, {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

bnds = ((0, None), (0, None))

res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds, constraints=cons)

print(res)

res2 = minimize_ipopt(fun, (2, 0), bounds=bnds, constraints=cons)

print(res2)