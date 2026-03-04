import pandas as pd
import os
import json
from ttp_problem import create_ttp_instance, load_ttp_benchmark
from S5_baseline import run_s5
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
        
def summary(completed_runs, mode, ttp_file=None, best_s5=None):
    summary_rows = []
    for run_id in sorted(completed_runs):
        if mode == 'benchmark':
            mf = f'results/run_{run_id:02d}_{mode}_{ttp_file}_metrics.csv'
        else:
            mf = f'results/run_{run_id:02d}_{mode}_metrics.csv'

        if not os.path.exists(mf):
            continue

        df = pd.read_csv(mf)
        if df.empty:
            continue

        last  = df.iloc[-1] # final-generation values
        row = {
            'run_id': run_id,
            'best_G': last.get('best_G', float('nan')),
            'best_tct': last.get('best_tct', float('nan')),
            'best_Makespan': last.get('best_Makespan', float('nan')),
            'best_profit': last.get('best_profit', float('nan')),
            'best_imbalance': last.get('best_imbalance', float('nan')),
            'hypervolume': last.get('hypervolume', float('nan')),
            'mean_genotypic_diversity': df['genotypic_diversity'].mean()  if 'genotypic_diversity'  in df.columns else float('nan'),
            'mean_phenotypic_diversity': df['phenotypic_diversity'].mean() if 'phenotypic_diversity' in df.columns else float('nan'),
            'elapsed_minutes': last.get('elapsed_time_seconds', float('nan')) / 60.0,
        }
        summary_rows.append(row)

    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)

        agg = {
            'mode':  mode,
            'n_runs': len(summary_df),
            'best_S5': best_s5 if best_s5 is not None else float('nan'),
        }
        
        col_rename = {
            'best_G': 'mean_best_G',
            'best_tct': 'mean_best_tct',
            'best_Makespan': 'mean_best_Makespan',
            'best_profit': 'mean_best_profit',
            'best_imbalance': 'mean_best_imbalance',
            'hypervolume': 'mean_hypervolume',
            'mean_genotypic_diversity': 'mean_genotypic_diversity',
            'mean_phenotypic_diversity': 'mean_phenotypic_diversity',
            'elapsed_minutes': 'mean_elapsed_minutes',
        }
        
        for src, dst in col_rename.items():
            agg[dst] = summary_df[src].mean() if src in summary_df.columns else float('nan')

        summary_out = pd.DataFrame([agg])
        if mode == 'benchmark':
            summary_file = f'results/summary_{mode}_{ttp_file}.csv'
        else:
            summary_file = f'results/summary_{mode}.csv'
        summary_out.to_csv(summary_file, index=False)
        
        
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
        # https://cs.adelaide.edu.au/~optlog/CEC2014COMP_InstancesNew/
        # Easy: eil51_n50_bounded-strongly-corr_01.ttp best known solution: 8200, good solutions: 4000, aceptable solutions: 2000
        # Medium: a280_n279_uncorr-similar-weights_05.ttp
        # Hard: kroA100_n990_uncorr_01.ttp or pr1002_n1001_uncorr_10.ttp
        ttp_file = 'eil51_n50_bounded-strongly-corr_01.ttp'        
        file = f"Instances/{ttp_file}"
        
        if os.path.exists(f'results/{ttp_file}_S5_best_fitness.txt'):
            with open(f'results/{ttp_file}_S5_best_fitness.txt', 'r') as f:
                best_s5_fitness = float(f.read().strip())
            print(f"S5 best fitness (G): {best_s5_fitness:.2f}")
        else:
            time_budget_seconds = 600 # 10 minutes        
            print(f"Running S5 baseline on {ttp_file} for {time_budget_seconds/60} minutes")
            best_s5_ind = run_s5(filepath=file, time_budget_seconds=time_budget_seconds, seed=42,)
            best_s5_fitness = -best_s5_ind.fitness  # G value (higher is better)
            print(f"S5 best fitness (G): {best_s5_fitness:.2f}")
            with open(f'results/{ttp_file}_S5_best_fitness.txt', 'w') as f:
                f.write(f"{best_s5_fitness:.2f}\n")
        problem = load_ttp_benchmark(file, mode=mode, seed=base_seed)
    else:
        best_s5_fitness = None
        # Create problem instance
        # Multi-robot Multi-objective TTP instance
        problem = create_ttp_instance(num_robots=num_robots, mode=mode, seed=base_seed)
    
    # Algorithm parameters
    params = {
        'pop_size': 100,
        'generations': 500,
        'cx_pb_tour': 0.8,
        'cx_pb_pack': 0.8,
        'mut_pb_tour': 0.2,
        'mut_pb_pack': 0.02,
        'tournament_size': 2,
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
                    'best_G': last_row.get('best_G'),
                    'best_tct': None if is_benchmark else last_row.get('best_tct'),
                    'best_Makespan': None if is_benchmark else last_row.get('best_Makespan'),
                    'best_profit': None if is_benchmark else last_row.get('best_profit'),
                    'best_scalar_fitness': None if is_benchmark else last_row.get('best_fitness'),
                    'final_diversity': last_row['genotypic_diversity'],
                    'total_time_seconds': last_row['elapsed_time_seconds'],
                    'generations': last_row['generation']
                }
                results.append(result)
                continue
        
        print(f"Running experiment {run_id+1}")
        # Run experiment
        
        run_single_experiment(base_seed + run_id, problem, params, metrics_file, checkpoint_file)
        
        # Mark as completed
        completed_runs.add(run_id)
        progress['completed_runs'] = sorted(list(completed_runs))
        progress['last_run'] = run_id
        save_progress(progress_file, progress)
        
    summary(completed_runs, mode, ttp_file if mode == 'benchmark' else None, best_s5=best_s5_fitness)

if __name__ == "__main__":
    main()