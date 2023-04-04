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


class NN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(1, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        
        self.test_layer = torch.nn.Linear(1, 1)

    def forward(self, x):

        return self.linear_relu_stack( x )



# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi*2.0, math.pi*2.0, 4000).reshape(-1,1)
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
#model = Polynomial3()
model = NN()

y_hat = model( x )

line2 = ax.plot( x, y_hat.detach().numpy() , '-r' )

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
for t in range(1000):
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
    





