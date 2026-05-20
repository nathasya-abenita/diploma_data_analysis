from evt_module import *

if __name__ == '__main__':
    # Example annual maxima
    # data = generate_annual_maxima(100)
    tp, gmst = read_annual_maxima() 

    # Fit
    params_hat_nonstat = gev_fit_gmst(tp, gmst) # mu0, sigma0, xi, alpha
    print(params_hat_nonstat)
    mu0, sigma0, xi, alpha = params_hat_nonstat

    #%% Create return level plot
    fig, ax = plt.subplots()

    # Present climate
    params_hat = reduce_params(gmst[-1], params_hat_nonstat)
    plot_empirical_return_level(transform_obs(tp, gmst, gmst[-1], params_hat_nonstat), ax=ax, color='tab:red')
    plot_fit_return_level(params_hat, ax=ax, label='present climate', color='tab:red')

    # Counterfactual
    params_hat = reduce_params(0, params_hat_nonstat)
    plot_empirical_return_level(transform_obs(tp, gmst, 0, params_hat_nonstat), ax=ax, color='tab:blue')
    plot_fit_return_level(params_hat, ax=ax, label='counterfactual climate', color='tab:blue')

    # Decorate plot
    decorate_return_level_plot(ax, activate_legend=True)

    # Plot all
    plt.show()