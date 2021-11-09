import torch
import matplotlib.pyplot as plt

def plot(x, y, *p):
    if type(x) == torch.Tensor:
        x = x.detach()
    if type(y) == torch.Tensor:
        y = y.detach()
    return plt.plot(x, y, *p)