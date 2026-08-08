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

    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

# 2. Write the training function
def train_neural_network(np_x_inputs: np.ndarray, np_y_outputs: np.ndarray):
   
   # convert from numpy to tensors
    x_tensor, y_tensor = convert_numpy_to_tensors(np_x_inputs,np_y_outputs)

    # initialize model, loss function, and optimizer
    model = PiecewiseLearner() 
    loss_func = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # adam optimizer for adaptive optimization
    epochs = 500
    # THE TRAINING LOOOOP (finally)
    for epoch in range(epochs + 1):
        # 1. Reset old gradients
        optimizer.zero_grad() #resets gradients from previous loop (if any)

        # 2. Forward pass
        y_hat = model(x_tensor)

        # 3. Compute Loss
        loss = loss_func(y_hat, y_tensor)

        #4. Backward pass (calculate gradients to figure out what the right direction to go in to minimize loss)
        loss.backward()

        # 5. Update the weights
        optimizer.step() #it nudges all the weights and stuff in the right direction here
        # The cool thing here is that because we are using the Adam optimizer, it will nudge weights that are having trouble learning with larger step sizes and the weights that are oscillating a lot would have smaller step sizes


        # Print Progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | MSE Loss: {loss}")

    return model

def convert_numpy_to_tensors(np_x_inputs: np.ndarray, np_y_outputs: np.ndarray):
    x_tensor = torch.tensor(np_x_inputs, dtype=torch.float32)
    y_tensor = torch.tensor(np_y_outputs, dtype=torch.float32)
    return x_tensor, y_tensor

