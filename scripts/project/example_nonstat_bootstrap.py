from evt_module import *

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