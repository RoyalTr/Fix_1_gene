import numpy as np
from numba import njit, types
from numba.typed import Dict, List
import time

start_time = time.time()

INPUT_FILE = 'results_data_per_generation.txt'
ATTEMPT_OUTPUT = 'aver_each_gen_per_attempt.txt'
SIM_OUTPUT = 'aver_each_gen_per_simulation.txt'

def parse_header(header_line):
    headers = header_line.strip().split(';')
    return {name: i for i, name in enumerate(headers)}

def load_data():
    with open(INPUT_FILE, 'r') as f:
        header = f.readline()
        idx = parse_header(header)
        
        raw_data = {}
        simul_param = {}

        col_indices = [idx[col] for col in [
            'N', 'freq_A', 'freq_Aa', 'freq_a', 'pan_homoz']]
        
        simul_indices = [idx[col] for col in [
            'Ni', 'r', 'K', 's_A', 'h_A', 'p_A_i', 'attempts']]
        
        for line in f:
            parts = line.strip().split(';')
            SimNr = int(parts[idx['SimNr']])
            attempt = int(parts[idx['attempt']])
            Rep = int(parts[idx['Rep']])
            generation = int(parts[idx['generation']])
            
            key = (SimNr, attempt, Rep)
            
            freq_data = [float(parts[i]) for i in col_indices]
            
            if key not in raw_data:
                raw_data[key] = []
            raw_data[key].append((generation, freq_data))
            
            simul_key = (SimNr, attempt)
            if simul_key not in simul_param:
                simul_param[simul_key] = [
                    int(parts[simul_indices[0]]),   # Ni
                    float(parts[simul_indices[1]]), # r
                    int(parts[simul_indices[2]]),   # K
                    float(parts[simul_indices[3]]), # s_A
                    float(parts[simul_indices[4]]), # h_A
                    float(parts[simul_indices[5]]), # p_A_i
                    int(parts[simul_indices[6]])    # attempts
                ]              
    
    return raw_data, simul_param

def complete_and_average_by_generation(raw_data, simul_param):
    result_rows = []
    
    attempt_groups = {}
    for (SimNr, attempt, Rep), entries in raw_data.items():
        key = (SimNr, attempt)
        if key not in attempt_groups:
            attempt_groups[key] = []
        attempt_groups[key].append((Rep, entries))
    
    for (SimNr, attempt), rep_list in attempt_groups.items():
        max_gen_attempt = 0
        rep_data = []
        
        for Rep, entries in rep_list:
            entries.sort(key=lambda x: x[0])
            max_gen_rep = entries[-1][0]
            last_vals = np.array(entries[-1][1])
            
            if max_gen_rep > max_gen_attempt:
                max_gen_attempt = max_gen_rep
            
            rep_data.append((Rep, entries, max_gen_rep, last_vals))
        
        gen_arrays = {gen: [] for gen in range(max_gen_attempt + 1)}
        
        for Rep, entries, max_gen_rep, last_vals in rep_data:
            gen_data = {gen: np.array(vals) for gen, vals in entries}
            
            for gen in range(max_gen_attempt + 1):
                if gen in gen_data:
                    gen_arrays[gen].append(gen_data[gen])
                else:
                    gen_arrays[gen].append(last_vals)
        
        for gen in range(max_gen_attempt + 1):
            values_array = np.array(gen_arrays[gen])
            avg = np.mean(values_array, axis=0)

            N_val = avg[0]
            freq_averages = avg[1:]
            
            meta = [SimNr, attempt] + simul_param[(SimNr, attempt)] + [gen, N_val]
            result_rows.append(meta + freq_averages.tolist())
    
    return result_rows

def compute_per_simulation_averages(attempt_rows):
    attempt_data = {}
    meta_map = {}
    
    for row in attempt_rows:
        SimNr = int(row[0])
        attempt = int(row[1])
        gen = int(row[9])
        N = float(row[10])

        vals = np.array([N] + row[11:])
        
        key = (SimNr, attempt)
        if key not in attempt_data:
            attempt_data[key] = {}
        
        attempt_data[key][gen] = vals
        
        if SimNr not in meta_map:
            meta_map[SimNr] = row[2:9]
    
    max_gen_per_sim = {}
    for (SimNr, attempt), gen_dict in attempt_data.items():
        max_gen_attempt = max(gen_dict.keys())
        if SimNr not in max_gen_per_sim:
            max_gen_per_sim[SimNr] = max_gen_attempt
        else:
            max_gen_per_sim[SimNr] = max(max_gen_per_sim[SimNr], max_gen_attempt)
    
    sim_data = {}
    for (SimNr, attempt), gen_dict in attempt_data.items():
        max_gen_attempt_avg = max(gen_dict.keys())
        max_gen_sim = max_gen_per_sim[SimNr]
        
        last_averaged_vals = gen_dict[max_gen_attempt_avg]
        
        for gen in range(max_gen_sim + 1):
            if gen not in gen_dict:
                gen_dict[gen] = last_averaged_vals
        
        if SimNr not in sim_data:
            sim_data[SimNr] = {}
        
        for gen in range(max_gen_sim + 1):
            if gen not in sim_data[SimNr]:
                sim_data[SimNr][gen] = []
            sim_data[SimNr][gen].append(gen_dict[gen])
    
    sim_rows = []
    for SimNr, gen_dict in sim_data.items():
        max_gen_sim = max(gen_dict.keys())
        
        for gen in range(max_gen_sim + 1):
            values_array = np.array(gen_dict[gen])
            avg = np.mean(values_array, axis=0)

            N_val = avg[0]
            freq_averages = avg[1:]
            
            row = [SimNr] + meta_map[SimNr] + [gen, N_val] + freq_averages.tolist()
            sim_rows.append(row)
    
    return sim_rows

def write_attempt_averages(rows):
    header = ('SimNr;attempt;Ni;r;K;s_A;h_A;p_A_i;attempts;generation;N;'
              'Ave_freq_A;Ave_freq_Aa;Ave_freq_a;Ave_pan_homoz\n')
    
    with open(ATTEMPT_OUTPUT, 'w') as f:
        f.write(header)
        for row in rows:
            row_str = [str(x) if i != 10 else f"{x:.4f}" for i, x in enumerate(row)]
            f.write(';'.join(row_str) + '\n')

def write_simulation_averages(rows):
    header = ('SimNr;Ni;r;K;s_A;h_A;p_A_i;attempts;generation;N;'
              'Ave_freq_A;Ave_freq_Aa;Ave_freq_a;Ave_pan_homoz\n')
    
    with open(SIM_OUTPUT, 'w') as f:
        f.write(header)
        for row in rows:
            row_str = [str(x) if i != 9 else f"{x:.4f}" for i, x in enumerate(row)]
            f.write(';'.join(row_str) + '\n')

def main():
    print("Loading raw data...")
    raw_data, meta1 = load_data()
    print("Averaging per generation per attempt...")
    attempt_averages = complete_and_average_by_generation(raw_data, meta1)
    print("Writing attempt-level averages...")
    write_attempt_averages(attempt_averages)
    print("Averaging per generation per simulation...")
    simulation_averages = compute_per_simulation_averages(attempt_averages)
    print("Writing simulation-level averages...")
    write_simulation_averages(simulation_averages)
    print("Done.")

if __name__ == '__main__':
    main()

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time required: {execution_time:.2f} seconds")