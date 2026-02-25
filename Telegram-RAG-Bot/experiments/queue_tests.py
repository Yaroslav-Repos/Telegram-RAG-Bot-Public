import numpy as np
import math


# ===============================
#  ARRIVAL PROCESS ESTIMATOR
# ===============================

def arrival_process_estimator(timestamps):
    """
    Оцінка інтенсивності λ.
    timestamps – список time.perf_counter() (секунди).

    Зберігаємо як інтервали у мс (для зручності),
    але λ рахуємо у запитах/сек.
    """
    if len(timestamps) < 2:
        return None

    diffs_sec = np.diff(sorted(timestamps))
    diffs_ms = diffs_sec * 1000.0

    mean_diff_ms = float(np.mean(diffs_ms))
    mean_diff_sec = mean_diff_ms / 1000.0

    lam = 1.0 / mean_diff_sec if mean_diff_sec > 0 else None

    return {
        "mean_interarrival_ms": mean_diff_ms,
        "mean_interarrival_sec": mean_diff_sec,
        "lambda": lam
    }


# ===============================
#  SERVICE TIME ESTIMATOR
# ===============================

def service_time_estimator(samples):
    """
    samples — service_time у секундах.
    Зберігаємо мс для кращого логування.
    μ — у 1/сек.
    """
    if not samples:
        return None

    mean_service_sec = float(np.mean(samples))
    mean_service_ms = mean_service_sec * 1000.0

    mu = 1.0 / mean_service_sec if mean_service_sec > 0 else None

    return {
        "mean_service_ms": mean_service_ms,
        "mean_service_sec": mean_service_sec,
        "mu": mu
    }

def build_qsystem_samples(arrival_log, latency_log, service_log, two_sided=False, eps=1e-6):
    """
    Побудова узгоджених вибірок для аналізу СМО на основі dict-логів.

    Вхід:
        arrival_log : dict[msg_id -> send_ts]
            Час надходження заявки (time.perf_counter() у секундах).
        latency_log : dict[msg_id -> latency_sec]
            Латентність виклику send_message для цієї заявки.
        service_log : dict[msg_id -> W_sec]
            Повний час W = resp_ts - send_ts (запит -> відповідь).

        two_sided : bool
            Якщо True, вважаємо, що мережна затримка приблизно симетрична
            і оцінюємо час обслуговування як:
                S_est = W - 2 * latency
            Якщо False, використовуємо:
                S_est = W - 1 * latency

        eps : float
            Мінімально допустиме значення для S_est, щоб уникнути нульових
            або від'ємних часів обслуговування.

    Повертає:
        dict з полями:
            "ids"         : список msg_id у порядку зростання arrival_ts
            "arrival_ts"  : список send_ts (для оцінки λ)
            "latencies"   : список latency_sec у тому ж порядку
            "W"           : список повних часів W = resp_ts - send_ts
            "S_est"       : список наближених часів обслуговування S_est

        або None, якщо спільних записів замало.
    """

    if not arrival_log or not latency_log or not service_log:
        return None

    # Спільні id, для яких є всі три вимірювання
    common_ids = set(arrival_log.keys()) & set(latency_log.keys()) & set(service_log.keys())
    if len(common_ids) < 2:
        return None

    # Сортуємо id за часом надходження заявки
    sorted_ids = sorted(common_ids, key=lambda mid: arrival_log[mid])

    arrivals = []
    latencies = []
    W_list = []
    S_list = []

    factor = 2.0 if two_sided else 1.0

    for mid in sorted_ids:
        ts = float(arrival_log[mid])
        lat = float(latency_log[mid])
        W = float(service_log[mid])

        net = factor * lat
        S = W - net
        if S <= 0.0:
            S = eps

        arrivals.append(ts)
        latencies.append(lat)
        W_list.append(W)
        S_list.append(S)

    return {
        "ids": sorted_ids,
        "arrival_ts": arrivals,
        "latencies": latencies,
        "W": W_list,
        "S_est": S_list,
    }

# ===============================
#  M/M/1 METRICS
# ===============================

def mm1_metrics(lmbd, mu):
    """
    Класична M/M/1 модель.
    """
    if not lmbd or not mu or lmbd <= 0 or mu <= 0:
        return {"error": "invalid parameters"}

    rho = lmbd / mu

    if rho >= 1.0:
        return {"error": "unstable system (rho >= 1 for M/M/1)"}

    # Формули M/M/1
    L = rho / (1.0 - rho)
    Lq = rho**2 / (1.0 - rho)
    W = L / lmbd
    Wq = Lq / lmbd

    return {
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq
    }

# =============================== # M/M/n METRICS # =============================== 
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
