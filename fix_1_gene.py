import numpy as np
import os
import sys
import time
import multiprocessing

# Gloals variables user can change
Repetitions = 20  # No. of times to rerun using the same parameter values. Max is hardware dependent.
generations = 100000000  # Prevent endless runs. Set to small nr. to view short initial trajectories or to debug.

# Prevent system sleep on Windows (when running overnight)
if os.name == 'nt':
    import ctypes
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

start_time = time.time()

def column_headings():
    return "Ni;r;K;s_A;attempts;h_A;p_A_i"

def results_headings():
    return "SimNr;Rep;Ni;r;K;N A fix;N a fix;s_A;attempts;h_A;p_A_i;Prob A fix;sd A fix;Aver A gen;sd A gen;Prob a fix;sd a fix;Aver a gen;sd a gen"

output_headings_avg = "SimNr;Reps;Ni;r;K;N A fix;N a fix;s_A;attempts;h_A;p_A_i;Prob A fix;sd A fix;Aver A gen;sd A gen;Prob a fix;sd a fix;Aver a gen;sd a gen"
results_filename_avg = "results_data_avg.txt"

example_rows = [
    "100;0.01;1000;0.001;1000;0.5;0.05",
    "1000;0.05;20000;0.005;20000;0.2;0.1"
]

input_filename = "input_data.txt"
headings = column_headings()

if not os.path.exists(input_filename):
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters to run in file input_data.txt (see example data), then rerun the program.")
    sys.exit(0)
    
with open(input_filename, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

if not lines:
    with open(input_filename, "w") as f:
        f.write(headings + "\n")
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters to run in file input_data.txt (see example data), then rerun the program.")
    sys.exit(0)

if lines[0] != headings:
    print(f"Error: The heading in input_data.txt is incorrect. Expected: '{headings}' but got '{lines[0]}'")
    print("Please correct the heading in input_data.txt to match the expected format.")
    sys.exit(0)

if len(lines) == 1:
    with open(input_filename, "a") as f:
        for row in example_rows:
            f.write(row + "\n")
    print("Please enter the parameters you want in file input_data.txt (see example data), then rerun the program.")
    sys.exit(0)

valid_data = []
error_found = False
for line_num, line in enumerate(lines[1:], start=2):
    parts = line.split(";")  # ';' is being used to delimit each variable
    if len(parts) != 7:
        print(f"Seven parameters must be found in file input_data.txt in line {line_num}. Please correct and rerun.")
        error_found = True
        continue
    try:
        Ni = int(parts[0]) # Initial population size, e.g., a founder group
        if not (1 <= Ni <= 1000000000):
            print(f"The value of Ni (initial population size) in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#1 The data in line {line_num} is wrong (Ni, initial population size). Please correct")
        error_found = True
        continue
    try:
        r = float(parts[1])
        if not (-10 <= r <= 10): # r for e.g. flies could be high, ca. 5
            print(f"The value of r (growth rate / generation) in line {line_num} must be between -10 and 10. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#2 The data in line {line_num} is wrong (r). Please correct")
        error_found = True
        continue
    try:
        K = int(parts[2]) # Maximum (carrying capacity) of the population
        if not (K >= Ni): # Carrying capacity must be >= initial population size
            print(f"The value of K (carrying capacity) in line {line_num} must be greater or equal to initial population size. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#3 The data in line {line_num} is wrong (K). Please correct")
        error_found = True
        continue
    try:
        s_A = float(parts[3])
        if not (-2 <= s_A <= 2):
            print(f"The value of s_A in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#4 The data in line {line_num} is wrong (s_A). Please correct")
        error_found = True
        continue
    try:
        attempts = int(parts[4])
        if not (1 <= attempts <= 1000000000000):
            print(f"The value of attempts in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#5 The data in line {line_num} is wrong (attempts). Please correct")
        error_found = True
        continue
    try:
        h_A = float(parts[5])
        if not (-1 <= h_A <= 1):
            print(f"#6 The value of h_A in line {line_num} is wrong. Please correct")
            error_found = True
            continue
    except ValueError:
        print(f"#7 The data in line {line_num} is wrong (h_A). Please correct")
        error_found = True
        continue
    try:
        p_A_i = float(parts[6])
        if not (0.0 <= p_A_i <= 1.0):
            print(f"#8 The value of p_A_i in line {line_num} must be between 0 and 1. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#9 The data in line {line_num} is wrong (p_A_i). Please correct")
        error_found = True
        continue

    valid_data.append((Ni, r, K, s_A, attempts, h_A, p_A_i))  # Scenario parameters to simulate

if error_found:
    print("Please correct the data and rerun the program")
    sys.exit(0)

def simulate_population(Ni, r, K, s_A, p_A_i, generations, attempts, h_A):
    # Initial counters for the beginning of each simulation run (affects each CPU running in parallel)
    a_count = 0  # Every time an A-allele is lost, an a-allele is fixed.
    A_count = 0  # Count all the times an A-allele fixes
    sum_A_fix_gens = 0.0      # Keep track of the total generations involved in A fixing.
    sum_A_fix_gens_sq = 0.0
    sum_a_fix_gens = 0.0 # Keep track of total generations for 'a' allele fixation.
    sum_a_fix_gens_sq = 0.0 # Keep track of squared generations for 'a' allele fixation.
    sum_N_A_final = 0.0 # Sum pop. size whenever A fixed over all attempts
    sum_N_a_final = 0.0 # Sum pop. size whenever a fixed over all attempts
    # Calculate fitness for each genotype
    fitness_AA = 1.0 + s_A
    fitness_Aa = 1.0 + h_A * s_A
    fitness_aa = 1.0
            
    # The simulation will see if an allele fixes, 'attempts' times
    for i in range(attempts):
        N = Ni  # Set population to initial size for every simulation attempt run.
        p_A_fix = 1 - (1 / (2 * N))  # When the proportion of A-allele p_A_t > p_A_fix then it has fixed
        p_A_t = p_A_i  # Set proportion of A allele to the initial proportion for each attempt run
        
        # Pre-calculate constant parts of Beverton-Holt equation below to avoid repeating calculation in the loop
        r_plus_1 = 1.0 + r
        r_div_K = r / K

        for gen in range(generations):
            if p_A_t == 0.0:  # Random segregation removed all A-alleles in the population.
                a_count += 1    # An a-allele fixed.
                sum_a_fix_gens += gen  
                sum_a_fix_gens_sq += gen * gen  
                sum_N_a_final += N # Add the pop size (N) across all N when a fixes.
                break
            elif p_A_t > p_A_fix:
                A_count += 1  # Increase the counter of A-allele fixed for 'attempts' times tried
                # If the current attempt to fix an allele succeeded add gen to sum_A_fix_gens
                sum_A_fix_gens += gen  # In Python 3 there is no upper limit for integers
                sum_A_fix_gens_sq += gen * gen
                sum_N_A_final += N # Add the pop size (N) across all N when A fixes.
                break
            
            # Calculate genotype frequencies
            freq_AA = p_A_t * p_A_t
            freq_Aa = 2.0 * p_A_t * (1.0 - p_A_t)  
            freq_aa = (1.0 - p_A_t) * (1.0 - p_A_t)

            # Calculate mean population fitness
            mean_fitness = freq_AA * fitness_AA + freq_Aa * fitness_Aa + freq_aa * fitness_aa

            # Calculate expected frequency of A alleles after selection
            numerator_A = 2.0 * freq_AA * fitness_AA + freq_Aa * fitness_Aa
            fit_A = numerator_A / (2.0 * mean_fitness)
            
            if r != 0:  # True if population is not fixed
                # N = round(N + r * N * (1 - N / K))  # Apply discrete population growth based on r and K
                N = round(N * r_plus_1 / (1.0 + r_div_K * N))  # Apply Beverton-Holt discrete population growth
                if N < 1: # Ensure N is at least 1
                    N = 1
                if N > K: # Ensure N does not exceed K
                    N = K
                p_A_fix = 1 - (1 / (2 * N))  # Prob. A is fixed changes as population size changes
            
            n_A_alleles = np.random.binomial(2 * N, float(fit_A))          
            p_A_t = n_A_alleles / (2 * N)          
        
    avg_N_A = sum_N_A_final / A_count if A_count > 0 else np.nan # Calculate the average N across times A fixed
    avg_N_a = sum_N_a_final / a_count if a_count > 0 else np.nan # Calculate the average N across times a fixed
    a_fix_prob = a_count / attempts
    A_fix_prob = A_count / attempts
    A_fix_sd = np.sqrt(A_fix_prob * (1.0 - A_fix_prob) / attempts)
    a_fix_sd = np.sqrt(a_fix_prob * (1.0 - a_fix_prob) / attempts)    
    # Calculate statistics for A allele fixation
    if A_count > 0:
        avg_A_fix_gen = sum_A_fix_gens / A_count
        variance_A = (sum_A_fix_gens_sq / A_count) - (avg_A_fix_gen * avg_A_fix_gen)
        std_A_fix_gen = np.sqrt(variance_A) if variance_A > 0 else 0.0
    else:
        avg_A_fix_gen = np.nan
        std_A_fix_gen = np.nan
        
    # Calculate statistics for 'a' allele fixation
    if a_count > 0:
        avg_a_fix_gen = sum_a_fix_gens / a_count  
        variance_a = (sum_a_fix_gens_sq / a_count) - (avg_a_fix_gen * avg_a_fix_gen)  
        std_a_fix_gen = np.sqrt(variance_a) if variance_a > 0 else 0.0  
    else:
        avg_a_fix_gen = np.nan
        std_a_fix_gen = np.nan
    
    return avg_N_A, avg_N_a, A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen, a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen

# Identify the parameters provided for each job and the data to be returned
def worker(job):
    idx, rep, Ni, r, K, s_A, attempts, h_A, p_A_i = job
    avg_N_A, avg_N_a, A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen, a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen = simulate_population(Ni, r, K, s_A, p_A_i, generations, attempts, h_A)
    return idx, rep, Ni, r, K, avg_N_A, avg_N_a, s_A, attempts, h_A, p_A_i, A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen, a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen

# Multiprocessing section.
if __name__ == '__main__':
    max_processes = multiprocessing.cpu_count()  # Find how many processors the executing CPU hardware can handle
    print(f"Maximum number of processes supported: {max_processes}")
    
    # jobs is an array of all the simulations that will be executed. If the input file has 5 sets 
    # of parameters  and repetitions = 20,then jobs will contain 100 parameter sets to run.
    jobs = []  
    for idx, (Ni, r, K, s_A, attempts, h_A, p_A_i) in enumerate(valid_data, start=1):
        for rep in range(1, Repetitions + 1):
            jobs.append((idx, rep, Ni, r, K, s_A, attempts, h_A, p_A_i))
    
    # Create a pool of worker processes (up to max_processes). Distributes the jobs among these processes, executing the
    # worker function for each job in parallel. The results variable will store the output from each worker call.
    # When a processor is done it is immediately given parameters from the new row in input_data.txt
    with multiprocessing.Pool(processes=max_processes) as pool:
        results = pool.map(worker, jobs)
    
    # Sort individual simulation results by SimNr x[0] and Reps x[1]).
    individual_results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
    
    # Group and average results over all the Repetitions for each simulation requested in input.data.txt.
    from collections import defaultdict
    grouped_results = defaultdict(list)
    
    # Group the sorted individual results. The loop iterates through each res (result) and appends it to the list associated with res[0]
    for res in individual_results_sorted:
        grouped_results[res[0]].append(res)
    
    results_by_param = []
    # Use list comprehensions. For a given group of results (which all share the same res[0] value), the next lines
    # extract specific data points from each individual result r.
    for idx, group in grouped_results.items():
        avg_N_A_list = [r[5] for r in group]
        avg_N_a_list = [r[6] for r in group]
        A_fix_prob = [r[11] for r in group]
        A_fix_sd = [r[12] for r in group]
        A_fix_gens = [r[13] for r in group]
        A_fix_sd_gens = [r[14] for r in group]
        a_fix_prob = [r[15] for r in group]
        a_fix_sd = [r[16] for r in group]
        a_fix_gens = [r[17] for r in group]
        a_fix_sd_gens = [r[18] for r in group]
        
        avg_A_fix_prob = sum(A_fix_prob) / len(A_fix_prob)
        avg_A_fix_sd = sum(A_fix_sd) / len(A_fix_sd)
        avg_A_fix_gens_val = np.nanmean(A_fix_gens)  # average only over successful fixations
        avg_std_A_fix_gens_val = np.nanmean(A_fix_sd_gens)  # average only over successful fixations!
        
        avg_a_fix_prob = sum(a_fix_prob) / len(a_fix_prob)
        avg_a_fix_sd = sum(a_fix_sd) / len(a_fix_sd)
        avg_a_fix_gens_val = np.nanmean(a_fix_gens)  
        avg_std_a_fix_gens_val = np.nanmean(a_fix_sd_gens)  
        
        Ni = group[0][2]
        r = group[0][3]
        K = group[0][4]
        avg_N_A = np.nanmean(avg_N_A_list)
        avg_N_a = np.nanmean(avg_N_a_list)
        s_A = group[0][7]
        attempts = group[0][8]
        h_A = group[0][9]
        p_A_i = group[0][10]

        results_by_param.append((idx, Ni, r, K, avg_N_A, avg_N_a, s_A, attempts, h_A, p_A_i,
                                  avg_A_fix_prob, avg_A_fix_sd, avg_A_fix_gens_val, avg_std_A_fix_gens_val,
                                  avg_a_fix_prob, avg_a_fix_sd, avg_a_fix_gens_val, avg_std_a_fix_gens_val))
    
    # Write individual simulation results to file "results_data.txt"
    results_filename = "results_data.txt"
    output_headings = results_headings()
    lines_to_write = []

    for rec in individual_results_sorted:
        line = f"{rec[0]};{rec[1]};{rec[2]};{rec[3]};{rec[4]};{rec[5]};{rec[6]};{rec[7]};{rec[8]};{rec[9]:.8f};{rec[10]:.8f};{rec[11]:.2f};{rec[12]:.2f};{rec[13]:.8f};{rec[14]:.10f};{rec[15]:.2f};{rec[16]:.2f};{rec[17]:.2f};{rec[18]:.2f}"
        lines_to_write.append(line)
    
    if os.path.exists(results_filename):
        with open(results_filename, "a") as f:
            for line in lines_to_write:
                f.write(line + "\n")
    else:
        with open(results_filename, "w") as f:
            f.write(output_headings + "\n")
            for line in lines_to_write:
                f.write(line + "\n")
    
    print(f"Results stored in file {results_filename}.")
    
    # If Repetitions > 1, write averaged results to file "results_data_avg.txt"
    if Repetitions > 1:
        # Check if the average results file exists. If not, create it and write headings.
        if not os.path.exists(results_filename_avg):
            with open(results_filename_avg, "w") as f:
                f.write(output_headings_avg + "\n")
        
        # Now append the new average results to the file.
        avg_lines = []
        for rec in results_by_param:
            line = (f"{rec[0]};{Repetitions};{rec[1]};{rec[2]};{rec[3]};{rec[4]:.2f};{rec[5]:.2f};"
                    f"{rec[6]};{rec[7]};{rec[8]:.8f};{rec[9]:.8f};{rec[10]:.10f};{rec[11]:.2f};"
                    f"{rec[12]:.2f};{rec[13]:.8f};{rec[14]:.10f};{rec[15]:.2f};{rec[16]:.2f};{rec[17]:.2f}")
            avg_lines.append(line)
            
        with open(results_filename_avg, "a") as f:
            for line in avg_lines:
                f.write(line + "\n")
        print(f"The average values based on the repetitions were stored in file: {results_filename_avg}")
            
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time required: {execution_time:.2f} seconds")