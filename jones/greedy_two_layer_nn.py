# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

"""The weight parameter ``model.fc2.weight`` is rescaled to a probability distribution.

``x_input`` is the input.

``y_output`` is the evaluation of $f(x)$.
"""

model = TwoLayerNet(1, 1000, 1)
model.fc2.weight = nn.Parameter(torch.abs(model.fc2.weight)/torch.sum(torch.abs(model.fc2.weight)))

x_inputs = torch.linspace(-1, 1, 100).unsqueeze(1).detach()
y_output = model(x_inputs).detach()

# uncomment if you need a plot.
#
# import matplotlib.pyplot as plt
# import numpy as np

# plt.plot(x_inputs.numpy(), y_output.numpy(), 'r')

"""# Implementation Part

## Task 1: Implement the following functions.

1. ``loss(y_output, h_output)``

2. ``get_alpha(y_output, h_output, sigma_output)``

"""

def loss(y_output, h_output):
    # This function implements the loss function as
    # \frac{1}{N}\sum_x |y(x) - h(x)|^2
    # where N is the number of sample points.

    # @param(y_output): the groundtruth data
    # @param(h_output): the approximation data

    # return the loss value
    pass

def get_alpha(y_output, h_output, sigma_output):
    # This function computes the best alpha to minimize
    #
    # loss(y_output, (1-alpha) * h_output + alpha * sigma_output)
    # which is consistent with the LaTeX equation: h_{k+1} = (1-alpha_k) * h_k + alpha_k * sigma(w_k * x + b_k)

    # @param(y_output)     is the output of the objective function on ``x_input``
    # @param(h_output)     is the output of the current approximation on ``x_input``
    # @param(sigma_output) is the output of a certain $\sigma(w x + b)$.

    # return alpha.
    pass


"""## Task 2: Implement the following function.

When solving

$$\arg\min_{\alpha, w, b}\|f(x) - \alpha h_k(x) - (1-\alpha) \sigma(w\cdot x + b)\|^2$$

You do not need to solve this exactly. As shown in the first question of HW2 (P3). You only need to find a pair of $(w, b)$ that

$$\langle f(x) - h_k(x), f(x) - \sigma(wx + b)\rangle < 0.$$

So, just sample enough ``nn.Linear(1,1)`` until the above inequality holds.
"""

def minimize_algorithm(x_inputs, y_output, n):
    # This algorithm finds a n-term shallow network to approximate y(x) at the sample points.

    # @param(x_inputs)   is the input x (needed to produce sigma_output).
    # @param(y_output)   is the output of the objective function on ``x_input``
    # @param(n)          is the number of terms (width) of the shallow network to be used.

    # return the final loss value for the n-term approximation.
    pass