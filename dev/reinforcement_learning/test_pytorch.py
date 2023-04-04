#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:52:17 2023

@author: alex
"""

import torch
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

show = True


class Polynomial3(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        #self.e = torch.nn.Parameter(torch.randn(()))
        #self.f = torch.nn.Parameter(torch.randn(()))
        #self.g = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        out = ( self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3 )
               #+ self.e * x ** 4 + self.f * x ** 5 + self.g * x ** 6 )
        
        return out

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'
    

class NN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )

    def forward(self, x):

        return self.linear_relu_stack( x )



period = 1
x = torch.linspace(-math.pi*period, math.pi*period, 4000).reshape(-1,1)
y = torch.sin(x).reshape(-1,1)

fig = plt.figure(figsize=(4,3), dpi=300)
ax = fig.add_subplot(111, autoscale_on=False )
ax.grid(True)
ax.tick_params( labelsize = 5)
ax.set_xlim(  [-math.pi*period,math.pi*period] )
ax.set_ylim(  [-1,1] )
line1 = ax.plot( x, y, '-b' )

if show: 
    fig.show()
    plt.ion()
    plt.pause( 0.001 )



model  = Polynomial3()
model2 = NN()

y_hat  = model( x )
y_hat2 = model2( x )

line2 = ax.plot( x, y_hat.detach().numpy() , '-r' )
line3 = ax.plot( x, y_hat2.detach().numpy() , '--g' )

criterion  = torch.nn.MSELoss(reduction='sum')
criterion2 = torch.nn.MSELoss(reduction='sum')

optimizer  = torch.optim.SGD(model.parameters(), lr=1e-6)
#optimizer2 = torch.optim.SGD(model2.parameters(), lr=1e-6)
optimizer2 = torch.optim.Adam( model2.parameters(), lr=1e-4 )

for t in range(5000):
    
    y_hat  = model(x)
    y_hat2 = model2(x)

    loss  = criterion(y_hat, y)
    loss2 = criterion2(y_hat2, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()
    
    if t % 100 == 99:
        print(t, loss.item())
        if show: 
            line2[0].set_data( x, y_hat.detach().numpy() )
            line3[0].set_data( x, y_hat2.detach().numpy() )
            plt.pause( 0.001 )





