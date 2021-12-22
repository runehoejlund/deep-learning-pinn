# %%
from torch.utils.data import Dataset
import torch
import numpy as np

func_types = ['poly', 'sin', 'abs', 'mix']
func_expressions = ['$1 + 2x + 3x^2 + 4x^3$', '$\sin 3 x$', '$|x|$', '$\exp(\sin 3x)$']
func_derivatives = ['$2 + 6 x + 12 x^2$', '$3 \cos 3 x$', 'sgn($x$)', '$3 \cos(3 x) \exp(\sin 3x)$']
func_titles = ['Polynomial', 'Trigonometric', 'Absolute', 'Mixed']

class DatasetFunction1D(Dataset):
    """1D function dataset."""

    def __init__(self, x, add_noise = False, func_type = 'poly'):
        super(Dataset, self).__init__()
        self.x = x
        self.add_noise = add_noise
        self.func_type = func_type

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]

        if self.func_type == 'poly':
            f = 1 + 2 * x + 3 * x**2 + 4 * x**3
            f_grad = 2 + 6 * x + 12 * x**2
        elif self.func_type == 'sin':
            f = torch.sin(3*x)
            f_grad = 3 * torch.cos(3*x)
        elif self.func_type == 'abs':
            f = torch.abs(x)
            f_grad = torch.sgn(x)
        elif self.func_type == 'mix':
            f = torch.exp(torch.sin(3*x))
            f_grad = 3*torch.cos(3*x)*torch.exp(torch.sin(3*x))
        
        if self.add_noise:    
            f = f + 0.2*np.std(f.detach().numpy())*torch.randn(f.shape).detach()
            f_grad = f_grad + 0.2*np.std(f_grad.detach().numpy())*torch.randn(f_grad.shape).detach()

        return (x, f, f_grad)

# %%
# Unit Tests
if __name__ == '__main__':
    import torch
    x = torch.linspace(-1,1, 100, requires_grad=True).reshape(-1,1)

    for func_type in ['poly', 'sin', 'abs', 'mix']:
        dataset = DatasetFunction1D(x, func_type=func_type)
        x, f, f_grad = dataset[:]
        assert len(dataset) == len(x)
        assert f.shape == x.shape