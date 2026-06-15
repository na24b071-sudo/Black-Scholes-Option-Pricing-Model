import math
from scipy.stats import norm

def black_scholes(S, K, T, r, vol, option_type = 'call'):
    d1 = (math.log(S/K) + (r + 0.5*(vol**2))*T) / (vol * math.sqrt(T))
    d2 = d1 - (vol * math.sqrt(T))

    if option_type.lower() == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    elif option_type.lower() == 'put':
        return K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        print('Invalid Option Type')

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    from scipy.optimize import brentq

    def objective(vol):
        return black_scholes(S, K, T, r, vol, option_type) - market_price

    return brentq(objective, 1e-6, 5)

def bs_delta(S, K, T, r, vol, option_type='call'):
    d1 = (math.log(S/K) + (r + 0.5 * vol**2)*T) / (vol * math.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1



#S: Underlying Price: If I go to market and want to buy one share of this stock I need to have this much money
#K: Strike Price: If I own the option I can buy the stock at this proce reagrless of its underlying price
#T: Time to Expiration: Option expires this much time into the future
#r: Risk-free Rate: 
#vol: Volatility