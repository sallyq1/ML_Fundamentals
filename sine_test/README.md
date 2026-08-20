The basic idea here is if I had no noise, sampled uniformly with a good density of points (no grid) and I trained a neural network on these points sampled from the sinusodial data distribution, then what would the next tanh function look like for different random seeds? 

--> different random seeds, same exact inputs
--> same random seeds, same inputs
--> same random seeds, different inputs (but sampled the same from the same distribution ofc)
--> different random seeds, different inputs


my guess is something like this W1* tanh(W1*tanh(x1_A*w1_A+b_A)+ W2_tanh(x1_B*w1_B+b_B) + B) + w2*tanh(W1*tanh(x1_A*w1_A+b_A)+ W2_tanh(x1_B*w1_B+b_B) + B) +B , etc etc etc for each neuron and layer