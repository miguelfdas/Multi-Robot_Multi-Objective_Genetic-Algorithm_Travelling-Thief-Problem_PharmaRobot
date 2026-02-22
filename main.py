import pandas as pd
import os
import json
from ttp_problem import create_ttp_instance, load_ttp_benchmark
from ga import GeneticAlgorithm

def load_progress(progress_file):
    # Load experiment progress
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'completed_runs': [], 'last_run': -1}

def save_progress(progress_file, progress):
    # Save experiment progress
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)

def run_single_experiment(seed, problem, params, metrics_file, checkpoint_file):
    # Create GA instance
    ga = GeneticAlgorithm(problem, seed=seed, **params)
    # Evolve
    ga.evolve(checkpoint_file=checkpoint_file)
    # Save metrics
    ga.save_metrics(metrics_file)
        
def main():
    
    mode = 'custom' # 'benchmark' or 'custom'
    
    # Configuration
    num_runs = 30
    base_seed = 42
    num_robots = 3
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load progress
    progress_file = 'results/progress.json'
    progress = load_progress(progress_file)
    completed_runs = set(progress['completed_runs'])
    
    if mode == 'benchmark':
        ttp_file = 'kroA100_n990_uncorr_01.ttp'
        file = f"Instances/{ttp_file}" 
        problem = load_ttp_benchmark(file, mode=mode, seed=base_seed)
    else:
        # Create problem instance
        # Multi-robot Multi-objective TTP instance
        problem = create_ttp_instance(num_robots=num_robots, mode=mode, seed=base_seed)
    
    # Algorithm parameters
    params = {
        'pop_size': 100,
        'generations': 500,
        'cx_pb': 0.8,
        'mut_pb_tour': 0.2,
        'mut_pb_pack': 0.002,
        'tournament_size': 3,
        'elitism': 2,
        'n_jobs': -1 # Use all available CPU cores for parallel evaluation
    }
    
    # Run experiments
    results = []
    for run_id in range(num_runs):

        if mode == 'benchmark':
            checkpoint_file = f'checkpoints/run_{run_id:02d}_{mode}_{ttp_file}.pkl'
            metrics_file = f'results/run_{run_id:02d}_{mode}_{ttp_file}_metrics.csv'
        else:
            checkpoint_file = f'checkpoints/run_{run_id:02d}_{mode}.pkl'
            metrics_file = f'results/run_{run_id:02d}_{mode}_metrics.csv'

        # Skip completed runs
        if run_id in completed_runs:        
            # Load existing results
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
                last_row = metrics_df.iloc[-1]
                is_benchmark = (mode == 'benchmark')
                result = {
                    'run_id': run_id,
                    'seed': base_seed + run_id,
                    'best_tct':            None if is_benchmark else last_row.get('best_tct'),
                    'best_Makespan':       None if is_benchmark else last_row.get('best_Makespan'),
                    'best_profit':         None if is_benchmark else last_row.get('best_profit'),
                    'best_scalar_fitness': None if is_benchmark else last_row.get('best_fitness'),
                    'final_diversity': last_row['genotypic_diversity'],
                    'total_time_seconds': last_row['elapsed_time_seconds'],
                    'generations': last_row['generation']
                }
                results.append(result)
        
        print(f"Running experiment {run_id+1}")
        # Run experiment
        
        run_single_experiment(base_seed + run_id, problem, params, metrics_file, checkpoint_file)
        
        # Mark as completed
        completed_runs.add(run_id)
        progress['completed_runs'] = sorted(list(completed_runs))
        progress['last_run'] = run_id
        save_progress(progress_file, progress)

if __name__ == "__main__":
    main()