import warnings
warnings.simplefilter('ignore')

%matplotlib inline
import seaborn
seaborn.mpl.rcParams['figure.figsize'] = (10.0, 6.0)
seaborn.mpl.rcParams['savefig.dpi'] = 90

import numpy as np
import pandas as pd
import pandas_datareader.data as web
try:
    ff=web.DataReader('F-F_Research_Data_Factors', 'famafrench')
except:
    ff=web.DataReader('F-F_Research_Data_Factors_TXT', 'famafrench')
ff = ff[0]

excess_market = ff.iloc[:,0]
ff.describe()

def sharpe_ratio(x):
    mu, sigma = 12 * x.mean(), np.sqrt(12 * x.var())
    values = np.array([mu, sigma, mu / sigma ]).squeeze()
    index = ['mu', 'sigma', 'SR']
    return pd.Series(values, index=index)
    
params = sharpe_ratio(excess_market)
type(params)