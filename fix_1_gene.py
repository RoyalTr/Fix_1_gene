import numpy as np
import os
import sys
import time
import multiprocessing
import warnings

warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)

# Global variables user can change
Repetitions = 20 # No. of times to rerun using the same parameter values. Max is hardware dependent.
generations = 100000000 # Prevent endless runs. Set to small nr. to view short initial trajectories or to debug.
document_results_every_generation = True # Set to True to output detailed per-generation data

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
results_filename_per_generation = "results_data_per_generation.txt"

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
    parts = line.split(";")
    if len(parts) != 7:
        print(f"Seven parameters must be found in file input_data.txt in line {line_num}. Please correct and rerun.")
        error_found = True
        continue
    try:
        Ni = int(parts[0])
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
        if not (-10 <= r <= 10):
            print(f"The value of r (growth rate / generation) in line {line_num} must be between -10 and 10. Please correct.")
            error_found = True
            continue
    except ValueError:
        print(f"#2 The data in line {line_num} is wrong (r). Please correct")
        error_found = True
        continue
    try:
        K = int(parts[2])
        if not (K >= Ni):
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

    valid_data.append((Ni, r, K, s_A, attempts, h_A, p_A_i))

if error_found:
    print("Please correct the data and rerun the program")
    sys.exit(0)

def simulate_population(Ni, r, K, s_A, p_A_i, generations, attempts, h_A):
    # Initial counters for the simulation run
    a_count = 0 # Count when a-allele fixes
    A_count = 0 # Count when A-allele fixes
    sum_A_fix_gens = 0.0 # Sum of generations for A fixation
    sum_A_fix_gens_sq = 0.0 # Sum of squared generations for A fixation
    sum_a_fix_gens = 0.0 # Sum of generations for a fixation
    sum_a_fix_gens_sq = 0.0 # Sum of squared generations for a fixation
    sum_N_A_final = 0.0 # Sum population size when A fixes
    sum_N_a_final = 0.0 # Sum population size when a fixes
        
    # Pre-calculate Beverton-Holt constants to increase performance
    r_plus_1 = 1.0 + r
    r_div_K = r / K

    # List to store per-generation data for all attempts
    per_generation_data = [] # Used if document_results_every_generation = True

    # Calculate fitness for each genotype
    fitness_AA = 1.0 + s_A
    fitness_Aa = 1.0 + h_A * s_A
    fitness_aa = 1.0
            
    # Run the simulation for all attempts
    for attempt_idx in range(1, attempts + 1):
        N = Ni # Reset population size
        p_A_fix = 1 - (1 / (2 * N)) # Definition of fixation threshold
        p_A_t = p_A_i # Reset A-allele proportion for each attempt

        for gen in range(generations):
            if document_results_every_generation:
                freq_a = 1.0 - p_A_t
                freq_Aa = 2.0 * p_A_t * freq_a
                pan_homoz = p_A_t ** 2 + freq_a ** 2
                per_generation_data.append((attempt_idx, gen, N, p_A_t, freq_a, freq_Aa, pan_homoz))

            if p_A_t == 0.0: # Check whether a-allele has fixed
                a_count += 1 # Sum every time a-allele fixed for that attempt to calc. statistics
                fixation_gen = gen
                sum_a_fix_gens += fixation_gen
                sum_a_fix_gens_sq += fixation_gen * fixation_gen
                sum_N_a_final += N
                break
            elif p_A_t > p_A_fix: # Check whether A-allele has fixed
                A_count += 1
                fixation_gen = gen
                sum_A_fix_gens += fixation_gen
                sum_A_fix_gens_sq += fixation_gen * fixation_gen
                sum_N_A_final += N
                break
    
            # Calculate genotype frequencies
            freq_AA = p_A_t * p_A_t
            freq_Aa = 2.0 * p_A_t * (1.0 - p_A_t)
            freq_aa = (1.0 - p_A_t) * (1.0 - p_A_t)

            # Calculate mean population fitness
            mean_fitness = freq_AA * fitness_AA + freq_Aa * fitness_Aa + freq_aa * fitness_aa

            # Calculate relative fitness of A alleles after taking selection into account
            # This is derived from frequency-dependent selection theory in population genetics.
            # Numerator: Each AA individual contributes 2 A alleles, weighted by the fitness of AA genotype
            # Each Aa individual contributes 1 A allele, weighted by the fitness of Aa genotype
            # Denominator: The factor of 2.0 accounts for diploid individuals (2 alleles per individual)
            numerator_A = 2.0 * freq_AA * fitness_AA + freq_Aa * fitness_Aa
            fit_A = numerator_A / (2.0 * mean_fitness)
    
            # Use Beverton-Holt model for logistic growth.
            # Use stochastic rounding to avoid N being stuck when N and r are very small.
            if r != 0:
                N_float = N * r_plus_1 / (1.0 + r_div_K * N)  # float average value
                frac = N_float - int(N_float)
                if np.random.random() < frac:
                    N = int(N_float) + 1  # Round up with the relevant probability
                else:
                    N = int(N_float)
    
            n_A_alleles = np.random.binomial(2 * N, float(fit_A))
            p_A_t = n_A_alleles / (2 * N)
    
    # Calculate statistics
    avg_N_A = sum_N_A_final / A_count if A_count > 0 else np.nan
    avg_N_a = sum_N_a_final / a_count if a_count > 0 else np.nan
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
        
    # Calculate statistics for a allele fixation
    if a_count > 0:
        avg_a_fix_gen = sum_a_fix_gens / a_count
        variance_a = (sum_a_fix_gens_sq / a_count) - (avg_a_fix_gen * avg_a_fix_gen)
        std_a_fix_gen = np.sqrt(variance_a) if variance_a > 0 else 0.0
    else:
        avg_a_fix_gen = np.nan
        std_a_fix_gen = np.nan
    
    return avg_N_A, avg_N_a, A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen, a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen, per_generation_data

def worker(job):
    sim_nr, rep, Ni, r, K, s_A, attempts, h_A, p_A_i = job
    # The worker function now returns the job parameters along with the simulation results,
    # so the main thread can correctly associate the results with the parameters.
    result = simulate_population(Ni, r, K, s_A, p_A_i, generations, attempts, h_A)
    return (sim_nr, rep, Ni, r, K, s_A, attempts, h_A, p_A_i, *result)

if __name__ == '__main__':
    max_processes = multiprocessing.cpu_count()
    print(f"Maximum number of processes supported: {max_processes}")
    
    if document_results_every_generation:
        if os.path.exists(results_filename_per_generation):
            os.remove(results_filename_per_generation)

        # Write the new heading to the file once
        with open(results_filename_per_generation, "w") as f:
            f.write("SimNr;attempt;Rep;Ni;r;K;s_A;h_A;p_A_i;attempts;generation;N;freq_A;freq_Aa;pan_homoz;freq_a\n")


    jobs = []
    all_raw_per_generation_results = []

    for sim_nr, (Ni, r, K, s_A, attempts, h_A, p_A_i) in enumerate(valid_data, start=1):
        for rep_idx in range(1, Repetitions + 1):
            jobs.append((sim_nr, rep_idx, Ni, r, K, s_A, attempts, h_A, p_A_i))

    with multiprocessing.Pool(processes=max_processes) as pool:
        results = pool.map(worker, jobs)

    individual_results_summary = []
    
    # Unpack the results tuple from the worker function.
    # The tuple now correctly contains all the original parameters.
    for res in results:
        (sim_nr, rep, Ni, r, K, s_A, attempts, h_A, p_A_i,
         avg_N_A, avg_N_a, A_fix_prob, A_fix_sd, avg_A_fix_gen,
         std_A_fix_gen, a_fix_prob, a_fix_sd, avg_a_fix_gen,
         std_a_fix_gen, per_generation_data) = res
        
        individual_results_summary.append((
            sim_nr, rep, Ni, r, K, avg_N_A, avg_N_a, s_A, attempts, h_A, p_A_i,
            A_fix_prob, A_fix_sd, avg_A_fix_gen, std_A_fix_gen,
            a_fix_prob, a_fix_sd, avg_a_fix_gen, std_a_fix_gen
        ))

        for entry in per_generation_data:
            all_raw_per_generation_results.append({
                'SimNr': sim_nr,
                'attempt': entry[0],
                'Rep': rep,
                'Ni': Ni,
                'r': r,
                'K': K,
                's_A': s_A,
                'h_A': h_A,
                'p_A_i': p_A_i,
                'attempts': attempts,
                'generation': entry[1],
                'N': entry[2],
                'freq_A': entry[3],
                'freq_a': entry[4],
                'freq_Aa': entry[5],
                'pan_homoz': entry[6]
            })
        
    if document_results_every_generation:
        # Sort by SimNr, then attempt, then Rep, and finally generation
        all_raw_per_generation_results_sorted = sorted(all_raw_per_generation_results,
                                                       key=lambda x: (x['SimNr'], x['attempt'], x['Rep'], x['generation']))
        with open(results_filename_per_generation, "a") as f_gen:
            for record in all_raw_per_generation_results_sorted:
                # Write the data row with pan_homoz included after freq_Aa
                f_gen.write(f"{record['SimNr']};{record['attempt']};{record['Rep']};{record['Ni']};{record['r']};{record['K']};{record['s_A']};{record['h_A']};{record['p_A_i']};{record['attempts']};{record['generation']};{record['N']};{record['freq_A']:.8f};{record['freq_Aa']:.8f};{record['pan_homoz']:.8f};{record['freq_a']:.8f}\n")
        print(f"Detailed per-generation results stored in file {results_filename_per_generation}.")

    individual_results_sorted = sorted(individual_results_summary, key=lambda x: (x[0], x[1]))
    
    from collections import defaultdict
    grouped_results = defaultdict(list)
    
    for res in individual_results_sorted:
        grouped_results[res[0]].append(res)
    
    results_by_param = []
    for idx, group in grouped_results.items():
        avg_N_A_list = [r[5] for r in group]
        avg_N_a_list = [r[6] for r in group]
        A_fix_prob = [r[11] for r in group]
        a_fix_prob = [r[15] for r in group]
        A_fix_gens = [r[13] for r in group]
        A_fix_sd_gens = [r[14] for r in group]
        a_fix_gens = [r[17] for r in group]
        a_fix_sd_gens = [r[18] for r in group]
        
        avg_A_fix_prob = sum(A_fix_prob) / len(A_fix_prob)
        avg_a_fix_prob = sum(a_fix_prob) / len(a_fix_prob)
        
        if len(A_fix_prob) > 1:
            variance_A_fix_prob = sum((p - avg_A_fix_prob) ** 2 for p in A_fix_prob) / (len(A_fix_prob) - 1)
            avg_A_fix_sd = np.sqrt(variance_A_fix_prob) if variance_A_fix_prob > 0 else 0.0
        else:
            avg_A_fix_sd = 0.0
        if len(a_fix_prob) > 1:
            variance_a_fix_prob = sum((p - avg_a_fix_prob) ** 2 for p in a_fix_prob) / (len(a_fix_prob) - 1)
            avg_a_fix_sd = np.sqrt(variance_a_fix_prob) if variance_a_fix_prob > 0 else 0.0
        else:
            avg_a_fix_sd = 0.0
        
        avg_A_fix_gens_val = np.nanmean(A_fix_gens)
        avg_std_A_fix_gens_val = np.nanmean(A_fix_sd_gens)
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
    
    results_filename = "results_data.txt"
    output_headings = results_headings()
    lines_to_write = []

    for rec in individual_results_sorted:
        line = f"{rec[0]};{rec[1]};{rec[2]};{rec[3]};{rec[4]};{rec[5]:.2f};{rec[6]:.2f};{rec[7]};{rec[8]};{rec[9]:.8f};{rec[10]:.8f};{rec[11]:.2f};{rec[12]:.2f};{rec[13]:.8f};{rec[14]:.10f};{rec[15]:.2f};{rec[16]:.2f};{rec[17]:.2f};{rec[18]:.2f}"
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
    
    if Repetitions > 1:
        if not os.path.exists(results_filename_avg):
            with open(results_filename_avg, "w") as f:
                f.write(output_headings_avg + "\n")
        
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
