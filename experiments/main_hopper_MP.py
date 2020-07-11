from torch import multiprocessing as mp



"""
1. Create a main where you create processes
2. create a learn() function where the agent samples and learns
    2.1 Sample step
        In the sample step we give the actor from the param manager to the sampler
    2.2 Learning step
        - In the learning step we instantiate a new a&c and copy the parameters from the a&c from the param manager
        - we calculate the loss for both a&c
        - we call the backward function on both a&c\
        - we copy the gradients from the a&c instance to the a&c from the param manager
        - we call optimizer.step() (Does this then also update the parameters from the pm a&c?)
        - we copy the parameters from the pm a&c to the average a&c  
        
3. 

"""