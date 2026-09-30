import numpy as np

def compute_pareto_front(self):
    """
    Extract non-dominated solutions (approximate Pareto front) from the current population
    A solution is non-dominated if no other solution is better or equal in all objectives and strictly better in at least one
    """
    
    objectives = np.array([ind.fitness for ind in self.population])
    
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= in all and < in at least one
            if (np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])):
                is_dominated[i] = True
                break
    
    self.pareto_front = objectives[~is_dominated]