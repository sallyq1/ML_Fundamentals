# Here we will sample the training data coming from the world we generated
import numpy as np

from plot import plot_points
from generate_world import get_y_outputs


noise_distribution_type = {
    1: "uniform",
    2: "skewed",
    3: "gaussian"
    # (add some other ones later)
}

sampling_distribution_type = {
    1: "uniform",
    2: "skewed"
    # (add some other ones later)
}

def generate_training_data(world_type: str, seed: int, number_of_datapoints: int, noise_dist_type: str, sampling_dist_type: str, noise_lvl: float): 
    # 1. initialize the random variable using seed
    rng = np.random.default_rng(seed)

    # 2. sample x inputs based on sampling distribution type

    if sampling_dist_type == "uniform": 
        # Samples uniformly between -5 and 5
        x_inputs = rng.uniform(-5, 5, size=(number_of_datapoints, 1))

    elif sampling_dist_type == "skewed": 
        # Exponential distribution creates a heavily right-skewed input space
        x_inputs = rng.exponential(scale=2.0,  size=(number_of_datapoints, 1))

    else: 
        raise ValueError(f"Unknown sampling type specification : {noise_dist_type}")

    # 3. Pass the clean inputs into the fake world function
    y_clean = get_y_outputs(world_type, x_inputs)


    # 4. Add noise to clean inputs
    if noise_dist_type == "gaussian":
        noise = rng.normal(loc=0.0, scale=noise_lvl, size=y_clean.shape)
    elif noise_dist_type == "uniform":
        # Noise bounded between [-noise_lvl, +noise_lvl]
        noise = rng.uniform(low=-noise_lvl, high=noise_lvl, size=y_clean.shape)
    elif noise_dist_type == "skewed":
        # Gumbel or Log-Normal distributions are great for skewed noise
        # This shifts a positive skew so its mean remains roughly 0
        noise = rng.gumbel(loc=0.0, scale=noise_lvl, size=y_clean.shape)
        noise -= noise.mean() 
    else:
        raise ValueError(f"Unknown noise distribution: {sampling_dist_type}")

    y_outputs = y_clean + noise
    print(x_inputs.shape, y_outputs.shape)
    plot_points(x_inputs, y_outputs)

    return x_inputs, y_outputs

def main():
    generate_training_data("world1", 24,  600, "uniform", "uniform", 0.6)
    generate_training_data("world2", 24,  1000, "uniform", "uniform", 0.3)


if __name__ == "__main__":
    main()

    