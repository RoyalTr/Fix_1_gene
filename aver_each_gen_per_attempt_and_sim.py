import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import time
from numba import jit
from numba.typed import List as NumbaList

start_time = time.time()

@jit(nopython=True)
def calculate_generation_averages_jit(freq_aa_values, max_generation):
    """
    JIT-compiled function to calculate generation averages.
    """
    result = np.zeros(max_generation + 1, dtype=np.float64)
    for gen in range(max_generation + 1):
        if gen < len(freq_aa_values):
            result[gen] = np.mean(freq_aa_values[gen])
        else:
            result[gen] = 0.0
    return result

@jit(nopython=True)
def calculate_attempt_averages_jit(rep_data_generations, rep_data_freq_aa, max_generation):
    """
    JIT-compiled function to calculate averages across repetitions for each generation.
    """
    # Initialize result array
    result = np.zeros(max_generation + 1, dtype=np.float64)
    
    # For each generation, collect freq_Aa values from all reps
    for gen in range(max_generation + 1):
        freq_aa_values = []
        
        # Go through all reps
        for rep_idx in range(len(rep_data_generations)):
            rep_generations = rep_data_generations[rep_idx]
            rep_freq_aa = rep_data_freq_aa[rep_idx]
            
            # Find if this rep has data for this generation
            found = False
            for i in range(len(rep_generations)):
                if rep_generations[i] == gen:
                    freq_aa_values.append(rep_freq_aa[i])
                    found = True
                    break
            
            if not found:
                freq_aa_values.append(0.0)
        
        # Calculate average for this generation
        if len(freq_aa_values) > 0:
            total = 0.0
            for val in freq_aa_values:
                total += val
            result[gen] = total / len(freq_aa_values)
        else:
            result[gen] = 0.0
    
    return result

@jit(nopython=True)
def calculate_simulation_averages_jit(attempt_results, max_generation, num_attempts):
    """
    JIT-compiled function to calculate averages across attempts for each generation.
    """
    result = np.zeros(max_generation + 1, dtype=np.float64)
    
    for gen in range(max_generation + 1):
        total = 0.0
        count = 0
        
        # Go through all attempts
        for attempt_idx in range(len(attempt_results)):
            attempt_data = attempt_results[attempt_idx]
            
            # Find data for this generation in this attempt
            found = False
            for i in range(len(attempt_data)):
                if i == gen and i < len(attempt_data):
                    total += attempt_data[i]
                    found = True
                    break
            
            if not found and gen < len(attempt_data):
                total += attempt_data[gen]
            elif not found:
                total += 0.0
            
            count += 1
        
        if count > 0:
            result[gen] = total / count
        else:
            result[gen] = 0.0
    
    return result

class SimulationDataProcessor:
    """
    A class to process simulation data with in-memory calculations,
    improved error handling, and JIT compilation for performance.
    """
    
    def __init__(self):
        self.raw_data = []
        self.simulation_params = {}
        self.attempt_averages = {}
        self.simulation_averages = {}
    
    def load_data(self, input_file: str) -> None:
        """
        Load and parse the raw simulation data file.
        """
        try:
            with open(input_file, 'r') as f:
                content = f.read()
            
            self._parse_raw_data(content)
            print(f"Successfully loaded data from {input_file}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {input_file} not found")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _parse_raw_data(self, content: str) -> None:
        """
        Parse raw simulation data from file content.
        """
        sim_blocks = content.strip().split('\n\n')
        
        for block in sim_blocks:
            if not block.strip():
                continue
            
            lines = block.strip().split('\n')
            simulation_data = self._parse_simulation_block(lines)
            
            if simulation_data:
                sim_nr = simulation_data['params']['sim_nr']
                self.simulation_params[sim_nr] = simulation_data['params']
                self.raw_data.extend(simulation_data['data'])
    
    def _parse_simulation_block(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parse a single simulation block.
        """
        sim_params_line = None
        sim_data_start = None
        
        # Find key lines
        for i, line in enumerate(lines):
            if line.startswith('SimNr;attempts;Ni;r;K;s_A;h_A;p_A_i'):
                sim_params_line = i
            elif line.startswith('SimNr;attempt;Rep;generation;N;freq_A;freq_a;freq_Aa'):
                sim_data_start = i
                break
        
        if sim_params_line is None or sim_data_start is None:
            return None
        
        # Parse parameters
        try:
            sim_params = lines[sim_params_line + 1].split(';')
            params = {
                'sim_nr': int(sim_params[0]),
                'attempts': int(sim_params[1]),
                'ni': int(sim_params[2]),
                'r': float(sim_params[3]),
                'k': int(sim_params[4]),
                's_a': float(sim_params[5]),
                'h_a': float(sim_params[6]),
                'p_a_i': float(sim_params[7])
            }
        except (ValueError, IndexError) as e:
            print(f"Error parsing simulation parameters: {e}")
            return None
        
        # Parse data rows
        data_rows = []
        for i in range(sim_data_start + 1, len(lines)):
            if lines[i].strip():
                try:
                    row = lines[i].split(';')
                    if len(row) >= 8:  # Now we have 8 columns including SimNr
                        data_rows.append({
                            'sim_nr': int(row[0]),  # SimNr from data row
                            'attempt': int(row[1]),
                            'rep': int(row[2]),
                            'generation': int(row[3]),
                            'N': int(row[4]),
                            'freq_A': float(row[5]),
                            'freq_a': float(row[6]),
                            'freq_Aa': float(row[7])
                        })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing data row: {e}")
                    continue
        
        return {'params': params, 'data': data_rows}
    
    def calculate_attempt_averages(self) -> None:
        """
        Calculate averages across repetitions for each attempt and generation.
        Optimized with JIT compilation for performance.
        """
        # Group data by simulation and attempt
        sim_attempt_data = defaultdict(lambda: defaultdict(list))
        
        for row in self.raw_data:
            sim_nr = row['sim_nr']
            attempt = row['attempt']
            sim_attempt_data[sim_nr][attempt].append(row)
        
        # Calculate averages for each simulation and attempt
        for sim_nr, attempts in sim_attempt_data.items():
            if sim_nr not in self.attempt_averages:
                self.attempt_averages[sim_nr] = {}
            
            for attempt, rows in attempts.items():
                # Group by rep within this attempt
                rep_data = defaultdict(list)
                for row in rows:
                    rep_data[row['rep']].append(row)
                
                # Find maximum generation and get N values
                max_generation, n_values = self._get_max_generation_and_n_values(rep_data)
                
                # Prepare data for JIT compilation
                rep_data_generations = []
                rep_data_freq_aa = []
                
                for rep_num, rep_rows in rep_data.items():
                    generations = np.array([r['generation'] for r in rep_rows], dtype=np.int32)
                    freq_aa = np.array([r['freq_Aa'] for r in rep_rows], dtype=np.float64)
                    rep_data_generations.append(generations)
                    rep_data_freq_aa.append(freq_aa)
                
                # Use JIT-compiled function for calculation
                if rep_data_generations:
                    avg_results = calculate_attempt_averages_jit(rep_data_generations, rep_data_freq_aa, max_generation)
                else:
                    avg_results = np.zeros(max_generation + 1, dtype=np.float64)
                
                # Create attempt results
                attempt_results = []
                for gen in range(max_generation + 1):
                    n_value = n_values.get(gen, 0)
                    
                    attempt_results.append({
                        'sim_nr': sim_nr,
                        'attempt': attempt,
                        'generation': gen,
                        'N': n_value,
                        'av_freq_Aa': avg_results[gen]
                    })
                
                self.attempt_averages[sim_nr][attempt] = attempt_results
    
    def calculate_simulation_averages(self) -> None:
        """
        Calculate averages across attempts for each simulation and generation.
        Optimized with JIT compilation for performance.
        """
        for sim_nr, attempts in self.attempt_averages.items():
            # Find the maximum generation across all attempts in this simulation
            max_generation = 0
            attempt_with_max_gen = None
            
            for attempt_num, attempt_results in attempts.items():
                attempt_max_gen = max(row['generation'] for row in attempt_results)
                if attempt_max_gen > max_generation:
                    max_generation = attempt_max_gen
                    attempt_with_max_gen = attempt_num
            
            # Get N values from the attempt with maximum generations
            n_values = {}
            if attempt_with_max_gen is not None:
                for row in attempts[attempt_with_max_gen]:
                    n_values[row['generation']] = row['N']
            
            # Prepare data for JIT compilation
            attempt_results_array = []
            for attempt_num, attempt_results in attempts.items():
                attempt_avg_values = np.array([row['av_freq_Aa'] for row in attempt_results], dtype=np.float64)
                attempt_results_array.append(attempt_avg_values)
            
            # Use JIT-compiled function for calculation
            if attempt_results_array:
                avg_results = calculate_simulation_averages_jit(attempt_results_array, max_generation, len(attempts))
            else:
                avg_results = np.zeros(max_generation + 1, dtype=np.float64)
            
            # Create simulation results
            simulation_results = []
            for gen in range(max_generation + 1):
                n_value = n_values.get(gen, 0)
                
                simulation_results.append({
                    'sim_nr': sim_nr,
                    'generation': gen,
                    'N': n_value,
                    'av_freq_Aa': avg_results[gen]
                })
            
            self.simulation_averages[sim_nr] = simulation_results
    
    def _get_max_generation_and_n_values(self, rep_data: Dict[int, List[Dict]]) -> Tuple[int, Dict[int, int]]:
        """
        Helper function to find maximum generation and N values.
        """
        max_generation = 0
        rep_with_max_gen = None
        
        for rep_num, rep_rows in rep_data.items():
            rep_max_gen = max(row['generation'] for row in rep_rows)
            if rep_max_gen > max_generation:
                max_generation = rep_max_gen
                rep_with_max_gen = rep_num
        
        # Get N values from the rep with maximum generations
        n_values = {}
        if rep_with_max_gen is not None:
            for row in rep_data[rep_with_max_gen]:
                n_values[row['generation']] = row['N']
        
        return max_generation, n_values
    
    def save_attempt_averages(self, output_file: str) -> None:
        """
        Save attempt averages to file.
        """
        with open(output_file, 'w') as f:
            for sim_nr in sorted(self.attempt_averages.keys()):
                params = self.simulation_params[sim_nr]
                
                # Write simulation parameters
                f.write("SimNr;attempts;Ni;r;K;s_A;h_A;p_A_i\n")
                f.write(f"{params['sim_nr']};{params['attempts']};{params['ni']};{params['r']};{params['k']};{params['s_a']};{params['h_a']};{params['p_a_i']}\n")
                
                # Write data header
                f.write("SimNr;attempt;generation;N;av_freq_Aa\n")
                
                # Write data for all attempts
                for attempt in sorted(self.attempt_averages[sim_nr].keys()):
                    for row in self.attempt_averages[sim_nr][attempt]:
                        f.write(f"{row['sim_nr']};{row['attempt']};{row['generation']};{row['N']};{row['av_freq_Aa']:.8f}\n")
                
                f.write("\n")
        
        print(f"Attempt averages saved to {output_file}")
    
    def save_simulation_averages(self, output_file: str) -> None:
        """
        Save simulation averages to file.
        """
        with open(output_file, 'w') as f:
            for sim_nr in sorted(self.simulation_averages.keys()):
                params = self.simulation_params[sim_nr]
                
                # Write simulation parameters
                f.write("SimNr;attempts;Ni;r;K;s_A;h_A;p_A_i\n")
                f.write(f"{params['sim_nr']};{params['attempts']};{params['ni']};{params['r']};{params['k']};{params['s_a']};{params['h_a']};{params['p_a_i']}\n")
                
                # Write data header
                f.write("SimNr;generation;N;av_freq_Aa\n")
                
                # Write data
                for row in self.simulation_averages[sim_nr]:
                    f.write(f"{row['sim_nr']};{row['generation']};{row['N']};{row['av_freq_Aa']:.8f}\n")
                
                f.write("\n")
        
        print(f"Simulation averages saved to {output_file}")
    
    def process_all(self, input_file: str, attempt_output: str, simulation_output: str) -> None:
        """
        Complete processing pipeline.
        """
        print("Starting data processing pipeline...")
        
        # Load and parse data
        self.load_data(input_file)
        print(f"Loaded {len(self.raw_data)} data rows from {len(self.simulation_params)} simulations")
        
        # Calculate averages
        print("Calculating attempt averages...")
        self.calculate_attempt_averages()
        
        print("Calculating simulation averages...")
        self.calculate_simulation_averages()
        
        # Save results
        print("Saving results...")
        self.save_attempt_averages(attempt_output)
        self.save_simulation_averages(simulation_output)
        
        print("Processing pipeline completed successfully!")

# Example usage
if __name__ == "__main__":
    processor = SimulationDataProcessor()
    
    input_file = "results_data_per_generation.txt"
    attempt_output = "aver_each_gen_per_attempt.txt"
    simulation_output = "aver_each_gen_per_simulation.txt"
    
    try:
        processor.process_all(input_file, attempt_output, simulation_output)
    except Exception as e:
        print(f"Error: {str(e)}")
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time required: {execution_time:.2f} seconds")