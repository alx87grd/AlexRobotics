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

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'
    




# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi*1.0, math.pi*1.0, 4000).reshape(-1,1)
y = torch.sin(x).reshape(-1,1)

fig = plt.figure(figsize=(4,3), dpi=300)
ax = fig.add_subplot(111, autoscale_on=False )
ax.grid(True)
ax.tick_params( labelsize = 5)
ax.set_xlim(  [-4,4] )
ax.set_ylim(  [-1,1] )
line1 = ax.plot( x, y, '-b' )

if show: 
    fig.show()
    plt.ion()
    plt.pause( 0.001 )


# Construct our model by instantiating the class defined above
model = Polynomial3()

y_hat = model( x )

line2 = ax.plot( x, y_hat.detach().numpy() , '-r' )

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_hat = model(x)

    # Compute and print loss
    loss = criterion(y_hat, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if t % 100 == 99:
        print(t, loss.item())
        if show: 
            line2[0].set_data( x, y_hat.detach().numpy() )
            plt.pause( 0.001 )
    


print(f'Result: {model.string()}')





