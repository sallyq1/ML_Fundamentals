The data that I used to generate the world used NumPy, to use Pytorch we'll need to switch to tensors

**Why?**  

    Pytorch uses a data structure called Tensors

**What is a tensor?** 

    Multi dimensional matrices that are great for rapid mathematical equations and gradient tracking

**What is the difference between a tensor and a numpy array?**

    1) Hardware: 
        - **Tensors:** Can run on CPUs, TPUs, and GPUs
        - **Numpy Array:**Only CPUs

    2) Use Case: 
        - **Numpy Array:** General-purpose scientific computing, data manipulation, and traditional machine learning
        - **Tensors:** Deep learning, building neural networks, and heavy multidimensional matrix math.