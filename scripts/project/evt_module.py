import numpy as np
import xarray as xr
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import genextreme, skew
import matplotlib.pyplot as plt

#%% Data generator

def read_annual_maxima():
    # Open file for Rx5day
    ds = xr.open_dataset("./data/era5_daily_tp/rx5day_sumatra.nc")

    # Take precipitation variable and convert unit
    pr = ds['tp'] * 1_000

    # Convert to DataFrame
    df = pr.to_dataframe().reset_index()[['valid_time', 'tp']]
    df = df[1:] # remove first row (shift excess)
    df['year'] = df['valid_time'].dt.year
    df = df.set_index(keys='year')

    # Read GMST data
    df_gmst = pd.read_csv(
        "./data/covariate/gmst.dat",
        comment="#",            # ignore lines starting with '#'
        sep=r'\s+',              # split on any whitespace
        header=None,            # no header row in the data
        names=["year", "gmst"]
        )
    df_gmst = df_gmst.set_index(keys='year')

    # Merge data to one DataFrame
    df = pd.merge(df, df_gmst, left_index=True, right_index=True)
    return df['tp'].array, df['gmst'].array

def generate_annual_maxima (n_years=50):

    # True GEV parameters
    mu_true = 120.0
    sigma_true = 25.0
    xi_true = 0.15

    # SciPy convention
    c_true = -xi_true
    rng = np.random.default_rng(42)
    x = genextreme.rvs(
        c=c_true,
        loc=mu_true,
        scale=sigma_true,
        size=n_years,
        random_state=rng
    )

    return x

#%% Likelihood functions

def gev_negloglik_gmst(params, data, gmst):
    """
    Negative log-likelihood for GEV distribution.

    Parameters
    ----------
    params : tuple
        (mu0, sigma0, xi, alpha)
    data : array-like
        Annual maxima data
    gmst : array-like
        GMST time series

    Returns
    -------
    float
        Negative log-likelihood
    """

    # Take parameters
    mu0, sigma0, xi, alpha = params

    # Apply model
    mu = mu0 * np.exp(alpha * gmst / mu0)
    sigma = sigma0 * np.exp(alpha * gmst / mu0)

    # Scale parameter must be positive
    if np.any(sigma <= 0):
        return np.inf

    z = (data - mu) / sigma
    t = 1 + xi * z

    # GEV support condition
    if np.any(t <= 0):
        return np.inf

    n = len(data)

    # Gumbel limit case
    if np.abs(xi) < 1e-6:
        ll = (
            - np.sum(np.log(sigma))
            - np.sum(z)
            - np.sum(np.exp(-z))
        )

    else:
        ll = (
            - np.sum(np.log(sigma))
            - np.sum( (1 + 1/xi) * np.log(t) )
            - np.sum(t ** (-1/xi))
        )

    return -ll

def gev_negloglik(params, data):
    """
    Negative log-likelihood for GEV distribution.

    Parameters
    ----------
    params : tuple
        (mu, sigma, xi)
    data : array-like
        Annual maxima data

    Returns
    -------
    float
        Negative log-likelihood
    """

    mu, sigma, xi = params

    # Scale parameter must be positive
    if sigma <= 0:
        return np.inf

    z = (data - mu) / sigma
    t = 1 + xi * z

    # GEV support condition
    if np.any(t <= 0):
        return np.inf

    n = len(data)

    # Gumbel limit case
    if np.abs(xi) < 1e-6:
        ll = (
            -n * np.log(sigma)
            - np.sum(z)
            - np.sum(np.exp(-z))
        )

    else:
        ll = (
            -n * np.log(sigma)
            - (1 + 1/xi) * np.sum(np.log(t))
            - np.sum(t ** (-1/xi))
        )

    return -ll

#%% General statistical commands

def sort_data(x):
    x_sorted = np.sort(x)

    n = len(x_sorted)

    # Weibull position
    p = np.arange(1, n + 1) / (n + 1)
    return x_sorted, p

def gev_cdf(z, params):
    mu, sigma, xi = params
    if np.abs(xi) < 1e-6:
        return np.exp( 
            -1.0 * np.exp(-1.0 * (z - mu) / sigma)
         )
    else:
        return np.exp( 
            -1.0 * (1 + xi * (z - mu) / sigma) ** (-1 / xi)
         )

def gev_ppf(p, params):
    mu, sigma, xi = params

    if np.abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(p))

    return (
        mu
        + (sigma / xi)
        * ((-np.log(p))**(-xi) - 1)
    )

# Plottings

def plot_qq(x, params, ax=None):

    # Sort data and compute probability
    x_sorted, p = sort_data(x)

    # Fitted quantiles
    q_fit = gev_ppf(p, params)

    # Prepare axis
    if ax == None:
        _, ax = plt.subplots(figsize=(6,6))

    # Add quantile points
    ax.scatter(x_sorted, q_fit)

    # 1:1 line
    mn = min(q_fit.min(), x_sorted.min())
    mx = max(q_fit.max(), x_sorted.max())
    ax.plot([mn, mx], [mn, mx], 'k--')

    # Labels
    ax.set_ylabel("Model")
    ax.set_xlabel("Empirical")
    ax.grid(True)
    return ax


def plot_fit_return_level (params, ax=None, label=None, color='tab:blue', rp_max=1_000):

    # Define return periods vector
    T = np.logspace(np.log10(1.01), np.log10(rp_max), 200)

    # Compute corresponding quantile
    zT = gev_ppf(1 - 1/T, params)

    if ax == None:
        _, ax = plt.subplots(figsize=(7,5))

    # Fitted curve
    ax.plot(T, zT, label=label, color=color)

    return ax

def plot_empirical_return_level (x, ax=None, label=None, color='tab:blue',
                                 alpha=1, marker='o'):

    # Sort data and find empirical return periods
    x_sorted, p = sort_data(x)
    T_emp = 1 / (1 - p)

    if ax == None:
        _, ax = plt.subplots(figsize=(7,5))

    # Empirical points
    ax.plot(T_emp, x_sorted, marker, label=label, color=color, alpha=alpha)
    return ax

def decorate_return_level_plot (ax, activate_legend=False):
    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Return Level")
    ax.grid(True)
    # ax.set_ylim(0, 500)
    if activate_legend:
        ax.legend()
    return ax

def plot_confidence_interval (ci_lower_params, ci_upper_params, ax=None, color='tab:blue',
                              label=r'$95\%$ confidence interval', rp_max=1_000):
    # Define return period vector
    T = np.logspace(np.log10(1.01), np.log10(rp_max), 200)

    # Define axis
    if ax == None:
        _, ax = plt.subplots(figsize=(7,5))

    # Compute quantile
    zT_lower = gev_ppf(1 - 1/T, ci_lower_params)
    zT_upper = gev_ppf(1 - 1/T, ci_upper_params)

    # Plot confidence interval
    ax.fill_between(T, zT_lower, zT_upper, 
                    color=color, alpha=0.15, linewidth=0, label=label)
    return ax

#%% Fittings

def gev_fit (data):
    # Initial guesses
    mu0 = np.mean(data)
    sigma0 = np.std(data)
    xi0 = -0.1
    initial = [mu0, sigma0, xi0]

    # Perform Maximum Likelihood Estimation
    result = minimize(
        gev_negloglik,
        initial,
        args=(data,),
        # method="L-BFGS-B",
        bounds=[
            (None, None),   # mu
            (1e-6, None),   # sigma > 0
            (-0.5, 1)         # xi bounds
        ]
    )

    # mu_hat, sigma_hat, xi_hat = result.x
    return result.x

def gev_fit_gmst (data, gmst):
    # Initial guesses
    mu0 = np.mean(data)
    sigma0 = np.std(data)
    xi0 = -0.1
    alpha0 = 0.1
    initial = [mu0, sigma0, xi0, alpha0]

    # Perform Maximum Likelihood Estimation
    result = minimize(
        gev_negloglik_gmst,
        initial,
        args=(data, gmst,),
        # method="L-BFGS-B",
        bounds=[
            (None, None),   # mu0
            (1e-6, None),   # sigma0 > 0
            (-0.5, 1),      # xi bounds
            (None, None)    # alpha
        ]
    )
    return result.x

def gev_fit_with_bootstrap (x, n_boot=1_000):

    # Define parameter list
    params_array = []

    # Resample and fit GEV
    for _ in range (n_boot):
        resample = np.random.choice(x, size=len(x), replace=True)
        params = gev_fit(resample)
        params_array.append(params)
    
    # Compute mean parameter estimates and confidence intervals
    params_array = np.array(params_array)
    mean_params = np.mean(params_array, axis=0)
    ci_lower_params, ci_upper_params = np.quantile(params_array, [0.025, 0.975], axis=0)
    
    return [mean_params, ci_lower_params, ci_upper_params]

def gev_fit_gmst_with_bootstrap (x, gmst, n_boot=1_000):

    # Define parameter list
    params_array = []

    # Resample and fit GEV
    for _ in range (n_boot):
        resample = np.random.choice(x, size=len(x), replace=True)
        params = gev_fit_gmst(resample, gmst)
        params_array.append(params)
    
    # Compute mean parameter estimates and confidence intervals
    params_array = np.array(params_array)
    mean_params = np.mean(params_array, axis=0)
    ci_lower_params, ci_upper_params = np.quantile(params_array, [0.025, 0.975], axis=0)
    
    return [mean_params, ci_lower_params, ci_upper_params]

# Define model
def reduce_params(gmst: float, params_hat: list):
    '''Reduce nonstationary GEV parameters back to stationary GEV'''

    # Update distribution parameters
    mu0, sigma0, xi, alpha = params_hat
    mu = mu0 * np.exp(alpha * gmst / mu0)
    sigma = sigma0 * np.exp(alpha * gmst / mu0)
    # Prepare output
    params_hat = [mu, sigma, xi]
    return params_hat

def transform_obs(tp_list: list, gmst_list: list, chosen_gmst: float, params_nonstat: list):
    tp_list_out = []
    for tp, gmst in zip(tp_list, gmst_list):
        params_hat_current = reduce_params(gmst, params_nonstat)
        params_hat_target = reduce_params(chosen_gmst, params_nonstat)
        p = gev_cdf(tp, params_hat_current)
        tp_new = gev_ppf(p, params_hat_target)
        tp_list_out.append(tp_new)
    return tp_list_out

def reduce_bootstrap_parameters(chosen_gmst, bootstrap_params):
    return [
        reduce_params(chosen_gmst, params) for params in bootstrap_params
    ]

if __name__ == '__main__':
    # Example annual maxima
    # data = generate_annual_maxima(100)
    tp, gmst = read_annual_maxima() 

    # Fit [mu0, sigma0, xi, alpha]
    bootstrap_params = gev_fit_gmst_with_bootstrap(tp, gmst) 

    #%% Create return level plot
    fig, ax = plt.subplots()

    # Present climate
    chosen_gmst = gmst[-1]
    color       = 'tab:red'
    bootstrap_params_reduced = reduce_bootstrap_parameters(chosen_gmst, bootstrap_params)
    plot_empirical_return_level(transform_obs(tp, gmst, chosen_gmst, bootstrap_params[0]), ax=ax, color=color)
    plot_fit_return_level(bootstrap_params_reduced[0], ax=ax, label='present climate', color=color)
    plot_confidence_interval(bootstrap_params_reduced[1], bootstrap_params_reduced[2], ax=ax, color='tab:red')

    # Decorate plot
    decorate_return_level_plot(ax, activate_legend=True)

    # Plot all
    plt.show()
