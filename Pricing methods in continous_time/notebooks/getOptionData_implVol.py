#%% # # # # # # # # # # # #GET PACKAGES
from yahoo_fin import options as op
import pandas as pd
import yfinance as yf
import datetime as dt

#%% # # # # # # # # # # # #Get Option Data 
def get_data(ticker):
    ticker = ticker

    expirationDates = op.get_expiration_dates(ticker)

    WholeCallData = pd.DataFrame()
    for _ in range(len(expirationDates)):
        WholeCallData = pd.concat([WholeCallData, op.get_calls(ticker, date = expirationDates[_])])
        
    WholeCallData['maturity_date'] = WholeCallData['Contract Name'].apply(extract_maturity)
    WholeCallData['maturity_date'] = pd.to_datetime(WholeCallData['maturity_date'])


    # date de last trade le plus frequent ! puis time to maturity
    specific_today = dt.datetime.now()


    WholeCallData['tau_to_matu_days'] = (WholeCallData['maturity_date'] - specific_today).dt.days
    WholeCallData['tau_to_matu_years'] = WholeCallData['tau_to_matu_days']/255
    
    WholeCallData['Mid'] = (WholeCallData["Bid"] + WholeCallData["Ask"])/2

    # Add new column with the converted values
    WholeCallData['impl_vol_yahoo_float'] = WholeCallData['Implied Volatility'].apply(convert_percentage_to_float)
    
    # Reformat
    WholeCallData = WholeCallData[["Contract Name",'maturity_date','tau_to_matu_days','tau_to_matu_years',"Strike", "Last Price", "Bid", "Ask", "Mid", "Change", "% Change", "Implied Volatility", 'impl_vol_yahoo_float']]
    WholeCallData.columns = ["Contract Name",'maturity_date','Maturity_days','Maturity_years',"Strike", "Last Price", "Bid", "Ask", "Mid", "Change", "% Change", "impl_vol_yahoo", 'impl_vol_yahoo_float']
    return WholeCallData

# Recompute maturity

def extract_maturity(option_code):
    year = option_code[4:6]
    month = option_code[6:8]
    day = option_code[8:10]
    return f"20{year}-{month}-{day}"

# Function to convert percentage string to float
def convert_percentage_to_float(percentage_str):
    return float(percentage_str.strip('%').replace(',', '')) / 100

