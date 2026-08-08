# Okay let's start with generating our fake world ...
# For now, we'll just focus on piecewise linear examples (because if you can't do that then what can you do really??)

# numpy is for heavy mathematical calculations
# matplotlib for drawing the actual graphics

from tkinter import Y
import numpy as np


def generate_fake_world1(x):

    #Fake world #1:
    # if X > 0  : -x^3 - 3x^2 + 2x + 1
    # if 0 <= X <= 2  :  cos(15x) * sin(5x^2) + 1
    # if X > 2  :  e^-(x-2) * cos(25x-50) + cos(30) * sin(20)

    # np.piecewise(input_variable, condition list, function list) 
    
    condition_list = [ x < 0, (0 <= x) & (x <= 2), x > 2 ]

    function_list = [ 
        lambda x: -x**3 - 3*x**2 + 2*x + 1, 
        lambda x: np.cos(15*x) * np.sin(5*x**2) + 1, 
        lambda x: np.exp(-(x-2)) * np.cos(25*x-50) + np.cos(30) * np.sin(20)
    ]

    return np.piecewise(x, condition_list, function_list)


def generate_fake_world2(x):

    #Fake world #2:
    # y = sin(2*pi*x) + 0.35*sin(6*pi*x) + (0.2*x**2)
    
    condition_list = [x > 0]

    function_list = [ 
        lambda x:  np.sin(2*np.pi*x) + 0.35*np.sin(6*np.pi*x) + 0.2*x**2
    ]

    return np.piecewise(x, condition_list, function_list)


_world_type_dict = {
    "world1": generate_fake_world1,
    "world2": generate_fake_world2
}

def get_y_outputs(world_type, x_inputs):
    # Get the y outputs from the x inputs
    y_outputs = _world_type_dict[world_type](x_inputs)
    return y_outputs
