import numpy as np 
from scipy.optimize import linprog 
from collections import defaultdict 
 
def v2(m: int) -> int: 
 if m == 0: 
 return 0 
 return (m & -m).bit_length() - 1 
 
def reduced_collatz(n: int) -> int: 
 if n == 1: 
 return 1 
 num = 3 * n + 1 
 return num // (1 << v2(num)) 
 
def V1(n: int) -> float: 
 if n == 1: 
 return 0.0 
 bits = n.bit_length() 
 low = 1 << (bits - 1) 
 high = 1 << bits 
 return min((n - low) / n, (high - n) / n) 
 
def V2(n: int) -> float: 
 if n == 1: 
 return 0.0 
 return 1.0 / (1 + v2(3 * n + 1)) 
 
def V3(n: int, M: float = 10.0) -> float: 
 if n == 1: 
 return 0.0 
 flog = n.bit_length() - 1 
 return flog / (M + flog) 
 
def V4(n: int) -> float: 
 return 1.0 if n % 3 == 0 else 0.0 
 
def vector_V(n: int) -> np.ndarray: 
 return np.array([V1(n), V2(n), V3(n), V4(n)]) 
 
def collect_constraints_by_v(max_n: int): 
 constraints = defaultdict(list) 
 for n in range(3, max_n + 1, 2): 
 v_val = v2(3 * n + 1) 
 constraints[v_val].append((vector_V(n), vector_V(reduced_collatz(n)))) 
 return constraints 
 
def solve_for_each_v(data, m=4): 
 solutions = {} 
 for v, points in data.items(): 
 A_ub = [] 
 b_ub = [] 
 for vn, vrn in points: 
 for i in range(m): 
 sum_other = np.sum(vn) - vn[i] 
 A_ub.append([-vn[i], -sum_other]) 
 b_ub.append(-vrn[i]) 
 A_ub.append([0, -1]); b_ub.append(0) 
 A_ub.append([-1, 1]); b_ub.append(0) 
 A_ub.append([-1, 0]); b_ub.append(0) 
 c = [1, m-1] 
 res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs') 
 if res.success: 
 alpha, beta = res.x 
 lambda_max = alpha + (m-1)*beta 
 solutions[v] = (alpha, beta, lambda_max) 
 return solutions 
 
def test_on_range(test_max, solutions, m=4): 
 violations = 0 
 total = 0 
 ratios = [] 
 w = np.ones(m) 
 for n in range(3, test_max + 1, 2): 
 v_val = v2(3 * n + 1) 
 if v_val not in solutions: 
 continue 
 alpha, beta, _ = solutions[v_val] 
 A = alpha * np.eye(m) + beta * (np.ones((m, m)) - np.eye(m)) 
 vn = vector_V(n) 
 vrn = vector_V(reduced_collatz(n)) 
 if (vrn > A @ vn + 1e-9).any(): 
 violations += 1 
 total += 1 
 S = np.dot(w, vrn) 
 Spred = np.dot(w, A @ vn) 
 if Spred > 1e-12: 
 ratios.append(S / Spred) 
 else: 
 ratios.append(0.0) 
 print(f"Tested {total} numbers, violations = {violations} ({100*violations/total:.4f}%)") 
 if ratios: 
 print(f"Weighted ratio: mean = {np.mean(ratios):.6f}, max = {np.max(ratios):.6f}") 
 return violations, total, ratios 
 
if __name__ == "__main__": 
 TRAIN_MAX = 500_000 
 TEST_MAX = 1_000_000 
 data = collect_constraints_by_v(TRAIN_MAX) 
 solutions = solve_for_each_v(data) 
 max_lambda = max(lam for _, _, lam in solutions.values()) 
 print(f"Worst-case λ = {max_lambda:.6f}") 
 test_on_range(TEST_MAX, solutions) 