
import pandas as pd
import numpy as np
import seaborn as sns
from numpy import matlib as mb

from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource

from datetime import datetime

#Get a workable dataset
import pandas_datareader.data as web

df = web.DataReader("SPY", 'yahoo', '2014-04-30', '2018-04-30') #get S&P500 index data
df = df.reset_index()  #reset index of the dataframe
df['Date'] = pd.to_datetime(df['Date']) #convertr Date to datetime

# ADF Test -> ADF Test tests for stationarity
from statsmodels.tsa.stattools import adfuller

adf_results = adfuller(df['Adj Close'])
print (adf_results) #ADF is -.47, so we cannot reject null hypothesis, meaning SPY plot is not stationary

#Creating an interactive HTML plot with Bokeh
data = ColumnDataSource(df)
p = figure(x_axis_type = 'datetime', plot_width = 300, plot_height = 300)
p.line('Date', 'Adj Close', line_width = 2, source = data)
show(p)  
#####################################################
# Hurst Exponent function -> Hurst exponent indicates whether the time series
#                  is stationary (H<0.5), random (H=0.5), or trending (H>0.5)

from numpy import log10, polyfit, var, subtract

def hurst_ernie_chan(p, lag_range=None):
    
    p_log = log10(p) # use log price

    variancetau = []
    tau = []
    
    # Create the range of lag values
    if lag_range == None:
        lags = [2]
    else:
        lags = range(2, lag_range) # lag_range < len(ts)

    for lag in lags: 
        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # call this pp or the price difference
        pp = subtract(p_log[lag:].values, p_log[:-lag].values)
        variancetau.append(var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = polyfit(log10(tau),log10(variancetau),1)
    hurst = m[0] / 2

    return hurst
 
print('Hurst exponent: {:.4}'.format(hurst_ernie_chan(df['Adj Close'], 5)))  #Hurst exponent of .458 indicates that the time-series is stationary  
#####################################################
# Variance Ratio
'''
Variance ratio can be used to test whether a financial return series is a pure random walk or having some predictability.
'''
def variance_ratio(ts, lag = 2):
    """
    Returns the variance ratio test result
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)
    
    # Apply the formula to calculate the test
    n = len(ts)
    mu  = sum(ts[1:n]-ts[:n-1])/n;
    m=(n-lag+1)*(1-lag/n);
    b=sum(np.square(ts[1:n]-ts[:n-1]-mu))/(n-1)
    t=sum(np.square(ts[lag:n]-ts[:n-lag]-lag*mu))/m
    return t/(lag*b);

vr = variance_ratio(df['Adj Close'], 5)
print(vr)
# vr of 1 equalts rejection of the random walk and vr of 0 means a series is a random walk
# in this case, vr of .91 means we can reject null hypothesis with 95% confidence

# Cointegration
# For a pair of assets, we can run a CADF test to determine the cointegration. 
# Let's see how to determine cointegration of two stocks: AAPL and IBM.
start = datetime(2014, 4, 30)
end = datetime(2018, 4, 30)

apple = web.DataReader("AAPL", 'yahoo', start, end) #Apple
ibm = web.DataReader("IBM", 'yahoo', start, end) #IBM

#Run cointegration test
import statsmodels.tsa.stattools as ts
merged = pd.merge(apple, ibm, on = 'Date')
x1 = merged['Adj Close_x']
y1 = merged['Adj Close_y']
cointegration = ts.coint(x1, y1)
print(cointegration)
# We found the test statistics is -1.19, greater than the 90% threshold -3.05,
# therefore, cannot reject the null hypothesis. That is, AAPL and IBM are not cointegration, as expected.

# Loading EP Chan's .mat data -> mat data are Matlab files
import scipy.io

mat = scipy.io.loadmat('inputData_ETF.mat') #load Matlab file

vol = pd.DataFrame(np.hstack((mat['tday'], mat['vol']))) # use `np.hstack` to make mini dataframe
cl = pd.DataFrame(np.hstack((mat['tday'], mat['cl'])))
lo = pd.DataFrame(np.hstack((mat['tday'], mat['lo'])))
hi = pd.DataFrame(np.hstack((mat['tday'], mat['hi'])))
data = pd.concat([vol, cl, lo, hi], keys=['vol', 'cl', 'lo', 'hi']) # use `pd.concat` to combine mini dataframes
syms = [item for sublist in np.array(mat['syms']) for items in sublist for item in items] # flatten list of lists of lists for symbol names
col_names = ['tday']+syms # prepare the col names for the final dataframe
data.columns = col_names # reset the column names
#to access MultiIndex, do data.xs('cl') for examples to get index of Close prices

# Converting 'tday' column into datetime type
temp = data['tday'].apply(lambda x: str(int(x))) # convert float to str
data['tday'].update(temp) # update dataset, now 'tday' is stored as str
temp2 = data['tday'].apply(lambda x: datetime.strptime(x, '%Y%m%d')) # convert str to datetime object
data['tday'] = temp2 # update dataset, now 'tday' is datetime object

# Plotting Close priecs of USO and GLD
data_close_2 = data.xs('cl')[['tday', 'USO', 'GLD']] # create a new df just for the 2 etfs

# create bokeh data object
close_for_plot = ColumnDataSource(data_close_2) # convert pd.DataFrame to bokeh.models.ColumnDataSource

# create a figure object
p = figure(x_axis_type="datetime", title="Closing Prices")
p.grid.grid_line_alpha=0.3
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Close Price'

# add elements to the figure
p.line('tday', 'USO', color='#A6CEE3', source=close_for_plot, legend='USO price')
p.line('tday', 'GLD', color='#FB9A99', source=close_for_plot, legend='GLD price')
p.legend.location = "top_left"

show(p)
#####################################################
# Developing a Bollinger bands strategy
'''
Bollinger bands is a practical mean reversion trading strategy. We enter the trade 
when price deviates entryZscore standard deviations from the mean. And we exit the
position when price mean-reverts to exitZscore standard deviations from the mean,
where exitZscore < entryZscore.

In this example, we set entryZscore=1 and exitZscore=0, which means we will exit
when the price mean-reverts to the current mean. Mean and standard deviation is
calcualted within the lookback period.

To develop a strategy, we first need to run a rolling OLS to determine the hedge
ratio between the two assets. Then, according to bollinger band definition, we find
the trading signal and update the dollar position of each asset class.
'''

import statsmodels.api as sm
y_ticker = 'USO'
x_ticker = 'GLD'

temp_y = data_close_2[y_ticker]
temp_x = data_close_2[x_ticker]

lookback = 20
entryZscore = 1.0
exitZscore = 0.0

data_close_2_copy = data_close_2 # copying original data, just in case
data_close_2_copy['hedge_ratio'] = np.nan #creating a nan column for future  hedge ratio storage

for t in range(lookback, len(temp_x)):  #loop through dates, for every data calculate hedge ratio using lookback period
    y = temp_y[t-lookback:t]
    x = temp_x[t-lookback:t]
    x = sm.add_constant(x)  #add constant for the statsmodels.api.OLS to work
    beta = sm.OLS(y, x).fit().params[1] #slope from OLS
    #dir(beta)
    data_close_2_copy.loc[t, 'hedge_ratio'] = beta   #store beta as hedge ratio
    
print (data_close_2_copy.shape)
print (data_close_2_copy.head(30))
###
# book page 71, use price spread `USO - hedgeRatio * GLD` as signal
cols = [x_ticker, y_ticker]
yport = np.ones(data_close_2_copy[cols].shape) #creating a array of 1's with same shape as data_close_2_copy columns
yport[:, 0] = -data_close_2_copy['hedge_ratio'] #assigning first column to be hedge ratio
yport = yport * data_close_2_copy[cols] #multiplying yport columns by data_close_2_copy columns 
         #('USO' column will be multiplied by 1's and 'GLD' column will be multiplied by the hedge_ratio)
yport = yport['GLD'] + yport['USO']  #addign 2 columns together to get one combined value

moving_avg = yport.rolling(window = lookback).mean()
moving_std = yport.rolling(window = lookback).std()
Zscore = (yport - moving_avg) / moving_std

# trade signals, boolean
long_entry = Zscore < -entryZscore # buy when price is lower than -entryZscore
long_exit = Zscore >= -exitZscore # sell when price is higher than or equal to -exitZscore
short_entry = Zscore > entryZscore # short buy when price is higher than entryZscore
short_exit = Zscore <= exitZscore # short sell when price is lower than or equal to exitZscore

num_units_long = np.empty((len(yport), 1))
num_units_long = pd.DataFrame(np.where(long_entry, 1, 0))
num_units_short = np.empty((len(yport), 1))
num_units_short = pd.DataFrame(np.where(short_entry, -1, 0))
num_units = num_units_long + num_units_short

# Create dollar position for each asset
temp1 = pd.DataFrame(mb.repmat(num_units, 1, 2)) #repeating num_units variable 1x2 times (twice)
temp2 = np.ones(data_close_2_copy[cols].shape)
temp2[:, 0] = -data_close_2_copy['hedge_ratio']

position = np.multiply(np.multiply(temp1, temp2), data_close_2_copy[cols]) #first we are multiplying dataframe
 # with positions by a -hedge ratio (thus GLD position will be -hedge ratio), then we multiplying result by asset
# price to compute how much of GLD we will be short when we are long 1 unit of USO (and vice versa)
position.columns = [x_ticker, y_ticker]

### Computing PnL and other performance metrics
#daily pnl in dollars
temp_3 = np.diff(data_close_2_copy[cols], axis=0) #daily price difference of GLD and USO
temp_4 = np.divide(temp_3, data_close_2_copy[cols][:-1]) #computing daily returns (Ptoday - Pyday) / Pydata
pnl = np.sum(np.multiply(position[:-1], temp_4), axis = 1) #multiplying positions by daily returns and summing them all together

# gross market value
mkt_value = pd.DataFrame.sum(abs(position[:-1]), axis=1) #combined value of positive and negative positions

# return is pnl divided by gross market value
ret = pnl / mkt_value
ret = ret.fillna(method = 'pad')  # 'pad' is the same as 'ffill'

# compute Sharpe
sharpe = (np.sqrt(252) * np.mean(ret)) / np.std(ret) 
APR = np.prod(1+ret) ** (252.0 / len(ret)) - 1
print('Price spread Sharpe: {:.4}'.format(sharpe))
print('Price spread APR: {:.4%}'.format(APR))

# Plot cumulative returns
acum_ret = ret.cumsum() # get cumulative return
acum_ret = acum_ret.fillna(method='pad') # fill with pad
acum_ret = acum_ret.fillna(0) # fill the rest NAN with 0

acum_ret_date=  pd.concat([data_close_2_copy['tday'][1:].reset_index(drop = True), acum_ret], axis = 1) # resetting
    # the index to concatenate both together based on the index (return data starts at 1, so we need to match indexes)
acum_ret_date.columns = ['tday', 'acum_ret'] # reset column names

acum_ret_for_plot = ColumnDataSource(acum_ret_date) # convert pd.DataFrame to bokeh.models.ColumnDataSource

p1 = figure(x_axis_type="datetime", title="Cumulative Returns")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Cumulative Returns'

p1.line('tday', 'acum_ret', color='#A6CEE3', legend='acum return', source=acum_ret_for_plot)
p1.legend.location = "top_left"

show(p1)
#####################################################
### Kalman Filter
'''
Instead of OLS, we can use Kalman filter to find the optimal hedge ratio and moving average. 
Kalman filter is a linear algorithm to update the expected value of hidden variable based on 
the latest value of an observable variable. In our case, y(t) = x(t)B(t)+ e(t),
the asset prices y is the observable variable, whereas B is the hidden variable. Note, B contains 
two parts, the slope and intercept, where slope represents the hedge ratio and intercept 
represents the moving average of the spread.
'''
from pykalman import KalmanFilter

data_kf = data[['tday', 'EWA', 'EWC']]
data_kf_cl = data_kf.xs('cl')
etfs = ['EWA', 'EWC']

def calc_slope_intercept_kalman(etfs, prices):
    
    delta = 1e-5
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.vstack([prices[etfs[0]], np.ones(prices[etfs[0]].shape)]).T[:, np.newaxis] #array
                             #of EWA (etf[0]) prices and a constant of 1
    
    kf = KalmanFilter(   #Kalman Filter calculation
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov
    )
    
    state_means, state_covs = kf.filter(prices[etfs[1]].values)  #applying Kalman filter with observation
                              #matrix from EWA (etf[0]) to EWC (etf[1])
    return state_means, state_covs  #state means represent B(t), with first column representing hedge ratio
                                    # and second column representing moving average

state_means, state_covs = calc_slope_intercept_kalman(etfs, data_kf_cl)  #state means represent B(t)
print (state_means)
###
# Plotting optimal hedge ratio and moving average of the spread
state_means_temp = pd.DataFrame(state_means)
state_means_df = pd.concat([data_kf_cl['tday'], state_means_temp], axis = 1)
state_means_df.columns = ['tday', 'hedge ratio', 'moving avg']

kf_for_plot = ColumnDataSource(state_means_df) # convert pd.DataFrame to bokeh.models.ColumnDataSource

p1 = figure(x_axis_type="datetime", title="Slope/Hedge Ratio")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Hedge Ratio'

p1.line('tday', 'hedge ratio', color='#A6CEE3', legend='slope/hedge ratio', source=kf_for_plot)
p1.line('tday', 'moving avg', color='#FB9A99', legend='intercept/moving avg', source=kf_for_plot)
p1.legend.location = "top_left"

show(p1)
#####################################################
### Cross-sectional mean reversion
'''
Cross-sectional mean reversion relies on the reversion of short-term relative returns of a basket 
of assets (usually stocks, not futures or currencies). The serial anti-correlation of the relative 
returns can generate profits. The relative return is the return of individual stock minus the 
average return of the stocks in a particular universe.
We determine the weights of every stock using: wi = -(r_i - <r_j>) / (sum from 1 to n(abs(r_every_basket_stock - <r_j>)))
where r_i is the daily return of ith stock, and <r_j> is the average daily return of all stocks in the index. 
If a stock has very positive return, we will short it, and if it has a very negative return, we will buy it.
'''
import scipy.io
import numpy as np
from datetime import datetime

mat = scipy.io.loadmat('inputDataOHLCDaily_20120504.mat')  #create a dictionary
#mat.keys()
syms = [item for sublist in np.array(mat['syms']) for items in sublist for item in items]
tday = pd.DataFrame(mat['tday'])
tday.columns = syms
cl = pd.DataFrame(mat['cl'])
cl.columns = syms

# Converting tday dataframe from type double to datetime.
def convert_df_to_datetime(df, datetime_format_str='%Y%m%d'):
    """Convert `double` to `datetime`.
    @param df: pd.DataFrame
    @param datetime_format_str: str, represents the format of datetime
    @return: pd.DataFrame
    """
    ncol = len(df.columns)
    for i in range(ncol):
        col_name = df.columns[i]
        df[col_name] = df[col_name].fillna(10000101.0) # for nan, use 1000-01-01 as a place holder
        temp = df[col_name].apply(lambda x: str(int(x))) # convert float to str
        df[col_name].update(temp) # update dataset, now `df` is stored as str
        temp2 = df[col_name].apply(lambda x: datetime.strptime(x, datetime_format_str)) # convert str to datetime object
        df[col_name] = temp2 # update dataset, now `df` is datetime object
    
    return df
        
def convert_series_to_datetime(series, datetime_format_str='%Y%m%d'):
    """Convert `double` to `datetime`.
    
    @param series: pd.Series
    @param datetime_format_str: str, represents the format of datetime
    @return: pd.Series
    """

    series = series.fillna(10000101.0) # for nan, use 1000-01-01 as a place holder
    temp = series.apply(lambda x: str(int(x))) # convert float to str
    series.update(temp) # update dataset, now `df` is stored as str
    temp2 = series.apply(lambda x: datetime.strptime(x, datetime_format_str)) # convert str to datetime object
    series = temp2 # update dataset, now `df` is datetime object
    
    return series

tday_datetime = convert_df_to_datetime(tday)

#getting 'cl' between start and end dates that all stocks have
start_day = '20080522'
end_day = '20120430'
start_mask = tday_datetime['AUD'] == start_day # use boolean mask to find index of a value within dataframe
end_mask = tday_datetime['AUD'] == end_day

idx_start = tday_datetime['AUD'][start_mask].index[0]
idx_end =  tday_datetime['AUD'][end_mask].index[0]

cl = cl.loc[idx_start:idx_end] # select range of index from "cl" dataframe

# Calculating Return and Weights
cl_lag = cl.shift(-1)
daily_return = np.divide(np.subtract(cl, cl_lag), cl_lag)
mkt_return = np.mean(daily_return, axis = 1)  #market return
mkt_return_list = np.tile(mkt_return, (daily_return.shape[1], 1)) # like repmat in *Matlab*, creates copies of an item
mkt_return_df = pd.DataFrame(mkt_return_list)
mkt_return_df = mkt_return_df.T
mkt_return_df.columns = syms
numerator = -(np.subtract(daily_return, mkt_return_df)) #subtracting market return from individual stock returns

denominator_temp = np.sum(np.abs(numerator), axis = 1)
denominator_list = np.tile(denominator_temp, (daily_return.shape[1], 1))
denominator_list = pd.DataFrame(denominator_list)
denominator = denominator_list.T
denominator.columns = syms
# can also do the same with pd.concat([denominator_temp]*daily_return.shape[1], axis = 1 )
weights = np.divide(numerator, denominator)
final_return = np.sum(np.multiply(weights.shift(1), daily_return), axis = 1) # shift weights downwards so that final return is calculated as yesterday's
acum_return = np.cumprod(1 + final_return) -1

# compute performance
sharpe = (np.sqrt(252) * np.mean(final_return)) / np.std(final_return)
APR = np.prod(1 + final_return) ** (252.0 / len(final_return)) - 1
print('Price spread Sharpe: {:.4}'.format(sharpe)) 
print('Price spread APR: {:.4%}'.format(APR))

#####################################################
### Trading calendar spread
'''
Futures usually do not show mean reversion. However, for a specific time period, 
currency pairs exist mean reversion. Therefore, we can use similar mean reversion 
strategy described above to trade currency cross-rates.
To do so, the first task is to identify a currency pair that is mean reverting. 
This has been discussed in Chen's book. Here we will try to recreate a mean 
reversion calendar spread strategy.
'''
import scipy.io
import numpy as np
from datetime import datetime

mat = scipy.io.loadmat('inputDataDaily_CL_20120813.mat') #load .mat file
contracts = [item for sublist in np.array(mat['contracts']) for items in sublist for item in items] # flatten list of lists of lists for symbol names
tday = pd.DataFrame(mat['tday'])
tday.columns = ['tday']
cl = pd.DataFrame(mat['cl'])
cl.columns = contracts

temp = pd.concat([cl, tday], axis = 1)
spot = temp[['tday', '0000$']]
contracts.remove('0000$')
cl = cl.drop('0000$', axis = 1)
#In this dataset, the first column is the spot price. The last column is the trading date. 
#The rest is different contracts. We first extract the spot price, pick a date and fit the 
#prices of futures of five nearest maturities to their time-to-maturity, T. In this way, we get gamma.
import statsmodels.api as sm

spot['gamma'] = np.nan
for t in range(spot.shape[0]): #this calculation takes time 
    ft = cl.iloc[t]
    idx = ft.reset_index().index #resetting index to get a range from 0 to n (instead of contract names as an index that is used)
    idx_list = idx.tolist()   #converting index range to a list
    idx_list_no_na = [idx_list[i] for i in range(len(ft)) if not ft.isna()[i]] # get rid of NAN ones - only vars with a non-NAN index
    idx_no_na = pd.Series(idx_list_no_na)  #converting list to a pandas Series
    idx_diff = idx_no_na.shift(-1) - idx_no_na  #creating a series of 1.0 values
    if len(idx_no_na) >= 5 and any(idx_diff[0:3]==1):
        ft_new = list(ft[idx_no_na[0:4]]) # use list instead of pd.Series so that sm.OLS can handle it
        tt = range(len(ft_new))
        tt = sm.add_constant(tt) # add constant for `statsmodels.api.OLS` to work
        beta = sm.OLS(ft_new,tt).fit().params[1] # return slope from `OLS`
        spot.loc[t, 'gamma'] = -12 * beta
print (spot['gamma'].tail())
''' TODO   1) Known gamma, we can find the half-life of it
 2) To apply mean reversion strategy, we need to find Zscore, with lookback set equal to the half-life
 3) A pair of contract is selected based on a set of criteria:
         First, the holding period for a pair of contracts is 3 months (61 trading days).
         Second, we roll forward to the next pair of contracts 10 days before the current near contract’s expiration.
         Third, the expiration dates of the near and far contracts are 1 year apart.
 Assume we hold a long position in the far contract, and a short contract in the near one initially, 
 we can find the true position according to our mean reversion strategy.
 4( Calculate the performance and plot cumulative return) 
'''
#####################################################
### Momentum Strategies
'''
The section aims to create a momentum strategy that buys and holds stocks within the 
top decile of 12-month lagged returns for a month, and vice versa for the bottom 
decile. In this way, we trade on a scross-sectional momentum of stocks.
As usual, the first task is to get the data in a workable format.
'''
import scipy.io
import numpy as np
from datetime import datetime

mat = scipy.io.loadmat('inputDataOHLCDaily_stocks_20120424.mat') #load .mat file (stocks from 2006-05-11 to 2012-04-24)
syms = [item for sublist in np.array(mat['stocks']) for items in sublist for item in items] # flatten list of lists of lists for symbol names
tday = pd.DataFrame(mat['tday'])
tday_datetime = convert_df_to_datetime(tday)
tday_datetime.columns = ['tday']
cl = pd.DataFrame(mat['cl'])
cl.columns = syms
# op = pd.DataFrame(mat['op'])
# op.columns = syms
print (cl.tail())
print (tday_datetime.tail())

# we can go ahead create the long and short signals. We calculate the return for 
# each day and sort the returns, and create a long and a short array respectively.
lookback = 252
holddays = 25
topN = 50
ret = (cl - cl.shift(lookback)) / cl.shift(lookback)

#Initialize signal matrix
longs = pd.DataFrame().reindex_like(cl) #returns dataframe with matching index to cl
shorts = pd.DataFrame().reindex_like(cl) #returns dataframe with matching index to cl

for t in range(lookback + 1, len(tday)): #calculation takes some time
    temp = ret.iloc[t] #panda series of returns for a particular day
    temp_sort = temp.sort_values() #sorting values, from smallest to largest
    idx = temp_sort.index  #index of stocks
    idx_list = idx.tolist()  #converting pandas Series to list
    idx_list_no_na = [idx_list[i] for i in range(len(temp_sort)) if not temp_sort.isna()[i]] #removing stocks wih NaN values
    longs.at[t, idx_list_no_na[-topN:]] = True  #.at is similar to .loc
    shorts.at[t, idx_list_no_na[0:topN]] = True
    
# Then, we create a zero positions matrix. We lag the long and short signal 
# according to our holding period. Update the positions matrix for every holding day.
positions = pd.DataFrame().reindex_like(longs)
positions = positions.fillna(0)    

for h in range(holddays):
    long_lag = longs.shift(h)
    long_lag = long_lag.fillna(0)
    long_lag = long_lag.astype('int')
    
    short_lag = shorts.shift(h)
    short_lag = short_lag.fillna(0)
    short_lag = short_lag.astype('int')
    
    positions = positions + long_lag.where(long_lag == 0, 1)
    positions = positions + short_lag.where(short_lag == 0 , -1)

# Once the positions matrix is obtained, we can trade accordingly. The daily and 
# cumulative return can be calculated. Sharpe ratio and APR can be obtained.
dailyret=np.sum(positions.shift(1) * (cl - cl.shift(1)) / cl.shift(1), axis=1) / (2 * topN) / holddays
dailyret.fillna(0)

# select specific start and end date
start_day = '20070515'
end_day = '20071231'
start_mask = tday_datetime['tday'] == start_day
end_mask = tday_datetime['tday'] == end_day
idx_start = tday_datetime['tday'][start_mask].index[0]
idx_end =  tday_datetime['tday'][end_mask].index[0]

# calculate cumulative return
cumret=np.cumprod(1 + dailyret[idx_start:idx_end]) - 1

sharpe = (np.sqrt(252) * np.mean(dailyret[idx_start:idx_end])) / np.std(dailyret[idx_start:idx_end]) 
APR = np.prod(1+dailyret[idx_start:idx_end]) ** (252.0 / len(dailyret[idx_start:idx_end])) - 1
print('Price spread Sharpe: {:.4}'.format(sharpe))
print('Price spread APR: {:.4%}'.format(APR))

# Plotting cumulative returns
acum_return_date = pd.concat([tday_datetime['tday'][idx_start:idx_end].reset_index(drop=True), cumret.reset_index(drop=True)], axis=1) # reset_index so that two df can be concatenated correctly
acum_return_date.columns = ['tday', 'acum_return'] # reset column names
acum_return_for_plot = ColumnDataSource(acum_return_date) # convert pd.DataFrame to bokeh.models.ColumnDataSource

p1 = figure(x_axis_type="datetime", title="Cumulative Returns")
p1.grid.grid_line_alpha=0.3
p1.xaxis.axis_label = 'Date'
p1.yaxis.axis_label = 'Cumulative Returns'

p1.line('tday', 'acum_return', color='#A6CEE3', legend='acum ret', source=acum_return_for_plot)
p1.legend.location = "top_left"

show(p1)
#####################################################
### End