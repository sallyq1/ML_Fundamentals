# Plot basic functions using matplotlib 

# NOTE: Later on let's work on figuring out how to use plotly for more complicated 3D-like functions (the fun stuff)


import numpy as np
import matplotlib.pyplot as plt
import torch

from generate_world import get_y_outputs

def plot_points(x_inputs, y_outputs):
    
    # 1. Create the display window (10 inches wide, 6 inches tall) 
    plt.figure(figsize=(10,6))

    # 2. Draw line graph on the grid we created
    plt.scatter(x_inputs, y_outputs, s=5)

    # 3. Show plot
    plt.show()

def plot_model_vs_world(model, world_type, x_train, y_train, start_window=-5, end_window=5, step_count = 1000):
    
    # sample the x values evenly
    x_inputs = np.linspace(start_window,end_window, step_count).reshape(-1,1)

    # get the y values for the true function (no noise)
    y_true = get_y_outputs(world_type, x_inputs)

    # get what the model predicts for those x values
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(x_inputs, dtype=torch.float32)).numpy()

    # now draw both on the same plot figure along with the scatter plot of the training data points
    plt.figure(figsize=(10,6))
    plt.scatter(x_train, y_train,s=5, color="gray", alpha=0.5) # plotting training points
    plt.plot(x_inputs,y_true, color="green", linewidth=2)
    plt.plot(x_inputs,y_pred, color="black", linewidth=2)

    plt.show()


