import numpy as np
import math

def mmn_metrics(lmbd, mu, n_servers): 
    """ M/M/n модель. λ – надходження (1/сек) 
    μ – обслуговування (1/сек) 
    n_servers – кількість паралельних серверів 
    """ 
    if not lmbd or not mu or lmbd <= 0 or mu <= 0: 
        return {"error": "invalid parameters"} 
    if n_servers <= 0 or not isinstance(n_servers, int): 
        return {"error": "n_servers must be positive integer"} 
    a = lmbd / mu 
    rho = lmbd / (n_servers * mu) 
    if rho >= 1.0: 
        return {"error": "unstable system (rho >= 1 for M/M/n)"}

    sum_terms = 0.0 
    for k in range(n_servers): 
        sum_terms += (a ** k) / math.factorial(k) 
    
    last_term = (a ** n_servers) / math.factorial(n_servers) * (1.0 / (1.0 - rho)) 
    p0 = 1.0 / (sum_terms + last_term)
    Pw = ((a ** n_servers) / math.factorial(n_servers)) * (1.0 / (1.0 - rho)) * p0 
    Lq = Pw * (rho / (1.0 - rho)) 
    L = Lq + a 
    Wq = Lq / lmbd 
    W = Wq + 1.0 / mu 
    return { 
        "n": n_servers, 
        "rho": rho, 
        "a": a, 
        "p0": p0, 
        "Pw": Pw, 
        "L": L, 
        "Lq": Lq, 
        "W": W, 
        "Wq": Wq }

print(mmn_metrics(8.15, 8.42, 10))
print(mmn_metrics(16.68, 8.42, 10))
print(mmn_metrics(23.85, 8.42, 10))

print(mmn_metrics(1.83, 0.1103, 50))
print(mmn_metrics(3.73, 0.1103, 50))
print(mmn_metrics(5.34, 0.1103, 50))

print(mmn_metrics(1.92, 7.21, 10))
print(mmn_metrics(3.94, 7.21, 10))
print(mmn_metrics(5.63, 7.21, 10))

print(0.11876484670581454*2.5+9.066183136907568*0.56+0.59*0.13869625520111512)
print(0.11876549901740786*2.5+9.069533868536562*0.56+0.59*0.13869625520111512)
print(0.11877731329915779*2.5+13.362006580905184*0.56+0.59*0.13869625520111512)
