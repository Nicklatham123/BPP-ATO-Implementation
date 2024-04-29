# %%
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
import time
import math
import re

class BinPacking:
    def __init__(self, capacity, weights, escape_local_minima):
        self.capacity = capacity
        self.weights = weights
        self.escape_local_minima = escape_local_minima

    def ato(self, problem, timeout):        
        solution = self.generate_random()
            
        best_fitness = self.evaluate(solution)
        iteration_results = [best_fitness]
        iterations_since_improvement = 0
        iters = 0
        elapsed_time = 0
        start_time = time.time()
        
        while elapsed_time < timeout:
            # Increase iterations
            iters += 1

            # Remove empty bins
            solution = list(filter(lambda bin:len(bin) > 0, solution))

            # Check for 1 bin
            if len(solution) > 1:
                # Random Selection of two atoms
                r1 = random.randint(0, len(solution)-1)
                r2_selection = [i for i in range(len(solution)) if i != r1]
                r2 = random.choice(r2_selection)
                
                ra1 = solution[r1]
                ra2 = solution[r2]
                
                # Collide 
                solution, iterations_since_improvement = self.collide(solution, ra1, ra2, iterations_since_improvement)
            
            # Check if fitness is more than best
            if self.evaluate(solution) < best_fitness:
                best_fitness = self.evaluate(solution)
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1
            iteration_results.append(best_fitness)
            elapsed_time = time.time() - start_time
        
        # Check if valid
        is_valid = self.check_valid(solution)
        print(f"Constraint Check: {is_valid}")
        return solution, iteration_results, iters

    def check_valid(self, solution):
        is_valid = True
        # Check that the sum of each bin is less or equal to the capacity
        for sublist in solution:
            sublist_sum = sum(sublist)
            # Set invalid if sum exceeds capacity
            if sublist_sum > self.capacity:
                print(f"The sum of {sublist} is {sublist_sum}, which exceeds {self.capacity}")
                is_valid = False
            else:
                continue
        return is_valid
        
    def generate_random(self):
        # Generate a initial random solution
        sol = []
        weights = deepcopy(self.weights)
        # While there are still items in the weights list
        while len(weights) > 0:
            # allocate the weight to a bin
            bin = []
            random_weight = weights[random.randint(0,len(weights)-1)]
            bin.append(random_weight)
            weights.remove(random_weight)
            sol.append(bin)
        return sol

    def evaluate(self, sol):
        # Remove empty bins from the solution
        no_empties = list(filter(lambda bin:len(bin) > 0, sol))
        sum_bin_fitness = 0
        for bin in no_empties:
            # Add the fitness of the bin to the sum 
            sum_bin_fitness += sum(bin)/self.capacity

        # Return the bin fitness
        average_bin_fitness = sum_bin_fitness/len(no_empties)
        fitness = len(no_empties) * (1 - average_bin_fitness)
        return fitness

    def evaluate_bin(self, bin):
        return self.capacity - sum(bin)         

    def collide(self, solution, a1, a2, iterations_since_improvement):
        new_solution = deepcopy(solution)
        original_solution = deepcopy(solution)
        
        a1_t2 = deepcopy(a1)
        a2_t2 = deepcopy(a2)

        # Measure if this iterations should be a detrimental one
        detriment = iterations_since_improvement > round(math.sqrt(len(solution))/2) and self.escape_local_minima
        
        if detriment:
            # Modify wants to be detrimental
            a1_wants = list(filter(lambda x:sum(a1_t2)*random.randint(2, 5) + x <= self.capacity, a2_t2))
            a2_wants = list(filter(lambda x:sum(a2_t2)*random.randint(2, 5) + x <= self.capacity, a1_t2))
        else:
            # Calculate wants as normal
            a1_wants = list(filter(lambda x:sum(a1_t2) + x <= self.capacity, a2_t2))
            a2_wants = list(filter(lambda x:sum(a2_t2) + x <= self.capacity, a1_t2))

        
        # Trade 1
        t1_before_score = self.evaluate_bin(a1_t2)
        if len(a1_wants) > 0:
            # Choose the max want
            a1_want = max(a1_wants)
            a1_t2.append(a1_want)
            a2_t2.remove(a1_want)
    
            new_solution[solution.index(a1)] = a1_t2
            new_solution[solution.index(a2)] = a2_t2
        t1_after_score = self.evaluate_bin(a1_t2)

        # Trade 2
        t2_before_score = self.evaluate_bin(a2)
        if len(a2_wants) > 0:
            # Choose the max want
            a2_want = max(a2_wants)
            a2.append(a2_want)
            a1.remove(a2_want)
    
            solution[solution.index(a1)] = a1
            solution[solution.index(a2)] = a2
        t2_after_score = self.evaluate_bin(a2)

        # Make comparisons and return the altruistically traded solution
        if detriment:
            iterations_since_improvement = 0
            if t1_after_score > t2_after_score:
                return new_solution, iterations_since_improvement
            elif t2_after_score > t1_after_score:
                return solution, iterations_since_improvement
            else:
                return original_solution, iterations_since_improvement
        else:  
            if t1_after_score < t2_after_score:
                return new_solution, iterations_since_improvement
            else:
                return solution, iterations_since_improvement

# %%
def plot_convergence(iteration_results, iters): 
    # Loop through iteration results
    for i, results in enumerate(iteration_results):        
        # Plot the convergence curve
        plt.plot(np.arange(1, iters+2), results, marker='o', linestyle='-', color='b', label=f'ATO Performance')

    # Plot variables
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title(f'Convergence Plot For ATO (num_collisions={iters}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
def run_testing(filename, runtime, test_runs):    
    # Read data from the file
    with open(f"bpp_datasets/{filename}", "r") as file:
        text = file.read()
    
    problems = []
    pattern = r"(\w+_\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+((?:\d+\s?)+)"
    matches = re.findall(pattern, text)

    # Use string analysis to extract problem from the dataset
    for match in matches:
        problem_name = match[0]
        capacity = int(match[1])
        num_items = int(match[2])
        best_known_solution = int(match[3])
        items = list(map(int, match[4].split()))
        problems.append({
            "problem_name": problem_name,
            "capacity": capacity,
            "num_items": num_items,
            "best_known_solution": best_known_solution,
            "items": items
        })

    # Loop through the first 6 problems
    for problem in problems[0:6]:
        # Set Problem Variables
        p_name = problem['problem_name']
        p_bks = problem['best_known_solution']
        p_capacity = problem["capacity"]
        p_num_items = problem["num_items"]
        p_items = problem["items"]

        # Set problem variables
        problem_performance_sum = 0
        problem_bin_sum = 0
        bins = 0
        performance = 0
        best_solution = float('+inf')
        
        # Loop test_runs times
        for i in range(test_runs):
            timeout = runtime
            
            # Run Algorithm
            s = BinPacking(p_capacity, p_items, True)
            start = time.time()
            solution, iteration_results, iters = s.ato(s, timeout)
            end = time.time()
    
            if len(solution) < best_solution:
                best_solution = len(solution)
                            
            # Write to file if solution exceeds best known
            if len(solution) < p_bks:
                print('Solution Exceeds Best Known')
                with open(f"bpp_results/{p_name}.txt", "a") as output_file:
                    output_file.write(f"Problem name: {p_name}\n")
                    output_file.write(f"Best Known Solution: {p_bks}\n")
                    output_file.write(f"Improved Solution: {len(solution)}\n")
                    output_file.write("Items:\n")
                    for item in solution:
                        output_file.write(f"{item}\n")
                    output_file.write("\n")
            
            print("Best Known Solution:", p_bks)
            print(f"Bins Used: {len(solution)}")
            # print(f"Time Elapsed: {round((end-start)*1000, 2)}ms")
            performance += p_bks / len(solution)
            bins += len(solution)
        
            # plot_convergence([iteration_results], iters)

        # Print Problem Set Performance
        avg_bins = bins / test_runs
        print(p_name)
        print(f'Average bins used: {avg_bins}')
        print(f"Best solution number of bins: {best_solution}" )
        print('\n')


# %%
# Usage
filename = 'bp_120.txt' # Set problem set file name
runtime = 3 # Set runtime in seconds
test_runs = 3 # Set number of test runs
run_testing(filename, runtime, test_runs)

# %%


# %%


# %%



