"""
DESCRIPTION
This code is an implementation of the Euler-Maruyama (EM) method for the numerical solution of a Stochastic Differential Equation (SDE). 
SDEs describe the dynamics that govern the time-evolution of systems subjected to deterministic and random influences.

AUTHOR
Hugo Michel
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from loguru import logger

def naive_GBM(
    drift: float,
    sigma: float,
    duration:float,
    dt: float, 
    n_dim = 1,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    
    n_steps: int=int(np.ceil(duration / dt))
    logger.info(f"number of steps for duration {duration} and dt {dt}: {n_steps}")
    dw: np.ndarray = np.random.normal(0.0, np.sqrt(dt), (n_steps, n_dim))
    result =  np.zeros((n_steps+1, n_dim))
    result[0, :] = initial_price * np.ones(n_dim)
    time_steps = np.linspace(0.0, duration + n_steps, n_steps+1)
    for step in tqdm(range(n_steps)):
        #Euler - Mayurama method
        dp = result[step] * (dt * drift + sigma * dw[step])
        result[step + 1] = result[step] + dp
    return pd.DataFrame(data=result, index=time_steps)


def analytical_solution_GBM(
    drift: float,
    sd: float,
    duration: float,
    dt: float,
    n_dim: int = 1,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    
    n_steps: int = int(np.ceil(duration / dt))
    logger.info(f"number of steps for duration {duration} and dt {dt}: {n_steps}")
    
    a = (drift - sd ** 2 / 2) * dt
    b = sd * np.random.normal(0, np.sqrt(dt), size=(n_steps, n_dim))
    x = np.exp(a+b)
    
    x = np.vstack([initial_price * np.ones(n_dim), x])
    
    x = x.cumprod(axis=0)
    
    time_steps = np.arange(0.0, duration + dt, dt)
    
    return pd.DataFrame(data = x, index = time_steps)

def analytical_solution_GBM_one_step(
    drift: float,
    sd: float,
    duration: float,
    dt: float,
    n_dim: int = 1,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    
    n_steps: int = int(np.ceil(duration / dt))
    logger.info(f"number of steps for duration {duration} : {1}")
    
    a = (drift - sd ** 2 / 2) * duration
    b = sd * np.random.normal(0, np.sqrt(duration), size=(1, n_dim))
    x = initial_price * np.exp(a+b)
    
    return pd.DataFrame(data=x)

def d1(k:float, t: float, sd: float, r: float, s: float) -> float:
    return (np.log(s / k) + (r + sd * sd / 2.0 / t)) / sd / np.sqrt(t)

def d2(k:float, t: float, sd: float, r: float, s: float) -> float:
    return d1(k, t, sd, r, s) - sd * np.sqrt(t)

def get_european_put_price(k: float, t: float, sd: float, r: float, s: float) -> float:
    return s * norm.cdf(d1(k, t, sd, r, s)) - np.exp(-r * t) * k * norm.cdf(d2(k, t, sd, r, s))


    