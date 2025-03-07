# Genetic Algorithm for Traveling Salesman Problem (TSP)

**Author: Masoud Rafiee**

[View detailed report](report.pdf)
---

## Techniques Used

### Crossover: Order Crossover (OX)
- **Description:** Selects a segment from one parent and fills remaining cities in order from the other parent.
- **Advantage:** Ensures a valid TSP path (each city visited exactly once) and maintains beneficial sequences.

**Example:** For a 13-city tour, cities between positions 3 and 7 from Parent1 directly copy to Child1. Remaining cities fill in from Parent2 in sequential order.

## Mutation Technique: Inversion Mutation

- Inversion mutation selects two positions and reverses the order of the cities between them.
- Helps escape local optima by significantly altering the tour sequence, maintaining permutation structure.

**Example:** Original segment `[A, B, C, D, E]` → Mutated segment `[A, E, D, C, B]`.

---

## GA Parameters
- **Generations:** 100
- **Population Sizes Tested:** 10, 20, 30, 40, 50, 60
- **Distance Metrics:** Euclidean and Manhattan

---

## Results

### Fitness Comparison:

| Metric     | Best Fitness | Best Path |
|------------|---------------|----------------------|
| Euclidean  | 188.23 | Iqaluit -> St. John's -> Fredericton -> Charlottetown -> Halifax -> Montreal -> Toronto -> Winnipeg -> Regina -> Victoria -> Edmonton -> Yellowknife -> Whitehorse -> Iqaluit |
| Manhattan | 260.28 | Whitehorse -> Yellowknife -> Edmonton -> Victoria -> Regina -> Winnipeg -> Toronto -> Montreal -> Fredericton -> Charlottetown -> Halifax -> St. John's -> Iqaluit -> Whitehorse |

- **Analysis:** Euclidean distance produced a shorter (optimal) path due to straight-line measurement, making it more realistic for TSP.

---

## Population Size Analysis
| Population Size | Best Fitness |
|-----------------|--------------|
| 10              | **211.07** |
| 20              | 218.55 |
| 30              | 224.93 |
| 40              | 219.89 |
| 50              | 219.58 |
| 60              | 221.35 |

- **Optimal Population Size:** 10 yielded the best performance, balancing quality and computational efficiency.
- **Larger populations:** Improved solution diversity but marginal benefits after a certain size (40-50).

---

## Conclusion
- **Best overall approach:** Order Crossover, Inversion Mutation, Euclidean distance.
- **Optimal fitness achieved:** 188.23 with Euclidean distance, population size 10.
- **Recommendation:** Use Euclidean distance for realistic TSP scenarios; maintain balanced population size for efficiency and quality.
