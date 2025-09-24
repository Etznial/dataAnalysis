import numpy as np
import random 


##### not used in the program, just left for future use
# The decay function (Probability Density Function)
# The area under this curve from x=0 to infinity is 1.
def decayFunc(x, tau=1):
    return 1/tau * np.exp(-x/tau)

# The Cumulative Distribution Function (CDF)
# For an exponential distribution, this is 1 - exp(-x/tau).
# This is what's used in the inverse CDF method.
def cdf(x, tau=1):
    return 1 - np.exp(-x/tau)

# The Inverse CDF (Inverse Cumulative Distribution Function)
# This function is used to convert uniformly distributed random numbers
# into exponentially distributed random numbers.
def invcdf(u, tau=1):
    return -tau * np.log(1 - u)
#####    



def run_simulation(sample_size, num_trials, tau=1):
    """
    Runs the Monte Carlo simulation to find the distribution of sample means.
    
    Args:
        sample_size (int): The number of random numbers in each sample.
        num_trials (int): The number of times to repeat the experiment.
        tau (int): The decay constant for the exponential distribution.
        
    Returns:
        list: A list of the mean values from each trial.
    """
    means = []
    
    # Use numpy for efficiency, as suggested by the professor.
    # We generate a matrix of random numbers and apply the invcdf function.
    # The 'size' argument creates a matrix of (sample_size, num_trials).
    # This is much faster than a nested loop.
    uniform_random_numbers = np.random.uniform(size=(sample_size, num_trials))
    exponential_samples = -tau * np.log(1 - uniform_random_numbers)
    
    # Calculate the mean of each column (each trial) and store the results.
    # The 'axis=0' argument tells numpy to compute the mean down the columns.
    means = np.mean(exponential_samples, axis=0)

    return means.tolist()



def mc(data, N=1):
    """
    Monte Carlo simulation for any data set that returns a data set of N * data.length() bigger.
    
    Args:
        data (list): The input data set.
        N (int): The factor by which to increase the data set size.
        
    Returns:
        list: The new, larger data set.
    """
    dSize = len(data)
    
    # Use np.random.choice to resample the data with replacement
    # The new size is N times the original size
    mcSim = np.random.choice(data, size=dSize * N, replace=True)
    
    return mcSim.tolist()




if __name__ == "__main__":
    # Parameters for the simulation
    sample_size = 10
    num_trials = 1000


    # Run the simulation and get the list of mean values
    fit_results = run_simulation(sample_size, num_trials)

    # Write the results to a file for gnuplot
    with open('mc.data', 'w') as file:
        for result in fit_results:
            file.write(f'{result}\n')


