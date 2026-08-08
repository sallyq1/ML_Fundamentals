import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



# 1. Define the architecture class 
class PiecewiseLearner(nn.Module):
    def __init__(self):
        super().__init__() # triggers initialization from pytorch
        
        #create the first hidden layer using a linear template (y = xW^T + b)
        self.hidden = nn.Linear(1, 64) # 1 input value per sample and 64 output features
        # projects that 1 one single value into 64 distinct nodes (each node learns its own unique slope and bias combination)
        # splits out input into 64 alternate perspectives
        
        self.relu = nn.ReLU() # instantiates the relu activation functiuon

        self.output = nn.Linear(64, 1) # the output dimension where it takes those 64 nodes and compresses them back into a single continous output (y_hat) prediction



def convert_numpy_to_tensors(np_x_inputs, np_y_outputs):
    x_tensor = torch.tensor(np_x_inputs, dtype=torch.float32)
    y_tensor = torch.tensor(np_y_outputs, dtype=torch.float32)

