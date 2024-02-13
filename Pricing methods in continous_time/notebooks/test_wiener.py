import numpy as np
import pandas as pd
from gbm import(
    naive_GBM,
    analytical_solution_GBM,
    analytical_solution_GBM_one_step,
    get_european_put_price,
)

import matplotlib.pyplot as plt

def test_naive_GBM(t = 1, dt = 1/250, initial_price = 100.0, n_dim = 1000, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False):
    prices = naive_GBM(
        drift = r, sigma = sd, duration = t, dt = dt, n_dim = n_dim
    )
    if plot_graph:
        prices.plot(legend=None)
        plt.title("Naive GBM (Euler Mayurama)")
        plt.xlabel("Maturity")
        plt.ylabel("Price")
        plt.show()
    mc = average_payoffs(prices, strike)
    print('Naive GBM price:', mc)
    return mc
    
def test_black_scholes(t = 1, dt = 1/250, initial_price = 100.0, n_dim = 1000, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False):
    print('Black-Scholes', get_european_put_price(k=strike, t=t, sd=sd, r=r, s=initial_price))
    return get_european_put_price(k=strike, t=t, sd=sd, r=r, s=initial_price)
    
def test_analytical_solution_GBM(t = 1, dt = 1/250, initial_price = 100.0, n_dim = 1000, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False):
    prices = analytical_solution_GBM(
        drift = r,
        sd=sd,
        duration=t,
        dt=dt,
        n_dim=n_dim,
        initial_price=initial_price,
    )
    if plot_graph:
        prices.plot(legend=None)
        plt.title("Analytical solution GBM")
        plt.xlabel("Maturity")
        plt.ylabel("Price")
        plt.show()
    mc = average_payoffs(prices, strike) 
    print('Analitycal solution GBM price:', mc)
    return mc
    
def test_analytical_solution_GBM_one_step(t = 1, dt = 1/250, initial_price = 100.0, n_dim = 1000, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False):
    prices = analytical_solution_GBM_one_step(
      drift = r,
      sd=sd,
      duration=t,
      dt=dt,
      n_dim=n_dim,
      initial_price=initial_price,
   )
    mc = average_payoffs(prices, strike) 
    print('Analitycal solution GBM 1-step price:', mc)
    return mc
   
def average_payoffs(prices, strike):
    final_prices = prices.iloc[-1]
    final_prices = np.array([final_prices - strike, np.zeros(final_prices.shape)])
    final_prices = np.max(final_prices, axis=0)
    mc = np.mean(final_prices)
    return mc

def price_over_nb_simul(nb_simul):
    list_naive_GBM_price = list()
    list_BS_price = list()
    list_analytical_solution_price = list()
    list_analytical_solution_one_step_price = list()
    
    for simul in nb_simul:
                
        price_GBM = test_naive_GBM(t = 1, dt = 1/250, initial_price = 100.0, n_dim = simul, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False)
        price_BS = test_black_scholes(t = 1, dt = 1/250, initial_price = 100.0, n_dim = simul, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False)
        price_analytical_GBM = test_analytical_solution_GBM(t = 1, dt = 1/250, initial_price = 100.0, n_dim = simul, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False)
        price_analytical_one_step_GBM = test_analytical_solution_GBM_one_step(t = 1, dt = 1/250, initial_price = 100.0, n_dim = simul, strike = 101.0, r = 0.02, sd = 0.1, plot_graph = False) 
        
        list_naive_GBM_price.append(price_GBM)
        list_BS_price.append(price_BS)
        list_analytical_solution_price.append(price_analytical_GBM)
        list_analytical_solution_one_step_price.append(price_analytical_one_step_GBM)
        

    columns = ["Naive GBM Price", "Black-Scholes Price", "Analytical GBM Price", "Analytical GBM 1-step Price", "Nb simulation"]
    
    # Creating a DataFrame
    df = pd.DataFrame({
        'Naive GBM Price': list_naive_GBM_price,
        'Black-Scholes Price': list_BS_price,
        'Analytical GBM Price': list_analytical_solution_price,
        'Analytical GBM 1-step Price': list_analytical_solution_one_step_price,
        'Nb simulation': nb_simul
    })
    
    plt.scatter(df["Nb simulation"], df["Naive GBM Price"], label="Naive GBM Price")
    plt.scatter(df["Nb simulation"], df["Black-Scholes Price"], label="Black-Scholes Price")
    plt.scatter(df["Nb simulation"], df["Analytical GBM Price"], label="Analytical GBM Price")
    plt.scatter(df["Nb simulation"], df["Analytical GBM 1-step Price"], label="Analytical GBM 1-step Price")
    plt.xlabel("Nombre simulation")
    plt.ylabel("Price")
    plt.title("Accuracy of price over the number of simulation")
    plt.legend()
    plt.show()
    
    df = df.round(2)
    
    print(df)
    
    df.to_csv("C:/Users/humic/OneDrive/Documents/Ecole/SorbonneFTD/Cours/Quantitative_Method_Finance/project_derivatives/prices.csv")
    
    return df
    

if __name__ == "__main__":
    
    print("#### NAIVE GBM (Euler-Mayurama) ####")
    test_naive_GBM()
    print("/n/n### BLACK-SCHOLES ###")
    test_black_scholes()
    print("/n/n### ANALYTICAL SOLUTION GBM ###")
    test_analytical_solution_GBM()
    print("/n/n### ANALYTICAL SOLUTION GBM ONE STEP PRICE ###")
    test_analytical_solution_GBM_one_step()
    
    print("#### Analyze price accuracy over nb_simuls ####")
    price_over_nb_simul(nb_simul)
    
    
    