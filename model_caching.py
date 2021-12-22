import torch
import torch.nn as nn
from BurgersNet import Net
import numpy as np
from pandas import DataFrame

def get_model_name(N_hid, N_layers, learning_rate, act_func, epochs):
    return ('net__N_hid_' + str(N_hid)
    + '__N_layers_' + str(N_layers)
    + '__act_fun_' + str(act_func)[:-2]
    + '__epochs_' + str(epochs)
    + '__lr_' + str(learning_rate))

def save_model(net: Net, losses, epochs, dir = './model/'):
    N_hid, N_layers, learning_rate, act_func, N_params = net.get_hyperparams()
    model_path = (dir + get_model_name(N_hid, N_layers, learning_rate, act_func, epochs))
    torch.save(net.state_dict(), model_path)
    torch.save(losses, model_path + '_losses.pt')

    losses = np.round([loss_data[-1].item(), loss_pde[-1].item(), loss_bc[-1].item(), loss_ic[-1].item()],5)
    results_df = DataFrame([[*losses, N_params, N_hid, N_layers, str(act_func)[:-2], epochs, learning_rate]], columns=['loss_data', 'loss_pde', 'loss_bc', 'loss_ic', 'N_params', 'N_hid', 'N_layers', 'act_fun', 'epochs', 'learning_rate'])
    results_df.to_csv(dir + get_model_name(N_hid, N_layers, learning_rate, act_func, epochs) + '.csv')

def load_model(N_hid = 30, N_layers = 3, learning_rate = 0.001, act_func = nn.Tanh(), epochs = 5000, dir = './model/'):
    '''
    @returns: model, (loss_data, loss_pde, loss_bc, loss_ic)
    '''
    model_path = (dir + get_model_name(N_hid, N_layers, learning_rate, act_func, epochs))
    model = Net(N_in=2, N_out=1, N_hid=N_hid, N_layers=N_layers)
    model.load_state_dict(torch.load(model_path))
    losses = torch.load(model_path + '_losses.pt')
    return model, losses