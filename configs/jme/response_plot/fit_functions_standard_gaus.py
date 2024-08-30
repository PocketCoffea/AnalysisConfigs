
# define the function to fit with 9 parameters
def std_gaus(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
    return (
        p0
        + p1 / (np.log10(x) ** 2 + p2)
        + p3 * np.exp(-p4 * (np.log10(x) - p5) ** 2)
        + p6 * np.exp(-p7 * (np.log10(x) - p8) ** 2)
    )


def fit_inv_median(ax, x, y, yerr, variable, y_pos, name_plot):
    print("fit_inv_median", variable)
    p_initial = [
        9.14823123e-01,
        1.59850801e00,
        1.08444406e01,
        -1.65510940e00,
        5.75460089e00,
        -4.40873200e-01,
        -4.04888813e-02,
        1.09142869e02,
        1.43155927e00,
    ]
    p_initial = [
        9.14823123e-01,
        1.59850801e00,
        1.08444406e01,
        -1.65510940e00,
        2.35460089e00,
        1.1,
        -4.04888813e-02,
        1.09142869e02,
        -1.43155927e00,
    ]
    p_initial = [
        1.4079260445453523,
        22.163070215571366,
        32.26551228077457,
        -1.0610367819625621,
        0.016828752007083572,
        0.46245487386104656,
        -4.375311791302624,
        18.346287800110574,
        0.8592087373356424,
    ]
    # do the fit
    popt, pcov = curve_fit(
        std_gaus, x, y, p0=p_initial, sigma=yerr, absolute_sigma=True
    )
    # popt, pcov = p_initial, [[0]*len(p_initial)]*len(p_initial)

    # plot the fit
    x_fit = np.linspace(x[0], x[-1], 1000)
    y_fit = std_gaus(x_fit, *popt)

    # print chi2 and p-value on the plot
    chi2 = np.sum(((y - std_gaus(x, *popt)) / yerr) ** 2)
    ndof = len(x) - len(popt)
    p_value = 1 - stats.chi2.cdf(chi2, ndof)

    # ax.plot(x_fit, y_fit, color=variables_plot_settings[variable][0], linestyle="--")
    # ax.text(
    #     0.98,
    #     0.2 + y_pos,
    #     f"$\chi^2$ / ndof = {chi2:.2f} / {ndof}" + f", p-value = {p_value:.2f}",
    #     horizontalalignment="right",
    #     verticalalignment="top",
    #     transform=ax.transAxes,
    #     color=variables_plot_settings[variable][0],
    #
    # )

    print(
        "\n",
        name_plot,
        "\nx",
        x,
        "\ny",
        y,
        "\nyerr",
        yerr,
        "\npopt",
        popt,
        "\npcov",
        pcov,
        "\nchi2/ndof",
        chi2,
        "/",
        ndof,
        "p_value",
        p_value,
    )
    fit_results = {
        "x": list(x),
        "y": list(y),
        "yerr": list(yerr),
        "parameters": list(popt),
        "errors": list(np.sqrt(np.diag(pcov))),
        "chi2": chi2,
        "ndof": ndof,
        "p_value": p_value,
    }

    return fit_results



def fit_inv_median_root(ax, x, y, xerr, yerr, variable, y_pos, name_plot):
    print("fit_inv_median_root", variable)

    # define the function to fit with 9 parameters
    # func_string = "[0] + [1] / (TMath::Log10(x) * TMath::Log10(x) + [2]) + [3] * TMath::Exp(-[4] * (TMath::Log10(x) - [5]) * (TMath::Log10(x) - [5])) + [6] * TMath::Exp(-[7] * (TMath::Log10(x) - [8]) * (TMath::Log10(x) - [8]))"
    func_string1 = "[0]+([1]/(pow(log10(x),2)+[2]))"
    func_string2 = "[0]+([1]/(pow(log10(x),2)+[2]))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))"
    func_string3 = "[0]+([1]/(pow(log10(x),2)+[2]))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))+([6]*exp(-([7]*((log10(x)-[8])*(log10(x)-[8])))))"
    func_root = ROOT.TF1("func_root", func_string2, x[0], x[-1])

    # func_root.SetParameters(
    #     9.14823123e-01,
    #     1.59850801e00,
    #     1.08444406e01,
    #     -1.65510940e00,
    #     2.35460089e00,
    #     1.1,
    #     -4.04888813e-02,
    #     1.09142869e02,
    #     -1.43155927e00,
    # )
    # func_root.SetParameters(
    #     0.9296687827807676,
    #     1.2461504555244276,
    #     7.595369858924967,
    #     -0.49346714415657444,
    #     23.577115065858514,
    #     1.199700717817504,
    #     -0.0404888813,
    #     116.04347276535667,
    #     -1.43155927,
    # )

    # func_root.SetParameter(0,-0.0221278)
    # func_root.SetParameter(1,119.265)
    # func_root.SetParameter(2,100)
    # func_root.SetParameter(3,-0.0679365)
    # func_root.SetParameter(4,2.82597)
    # func_root.SetParameter(5,1.8277)
    # func_root.SetParameter(6,-0.0679365)
    # func_root.SetParameter(7,3.82597)
    # func_root.SetParameter(8,1.8277)
    # func_root.SetParLimits(6,-20,10)
    # func_root.SetParLimits(7,0,100)
    # func_root.SetParLimits(3,-15,15)
    # func_root.SetParLimits(4,0,500)
    # func_root.SetParLimits(0,-2,25)
    # func_root.SetParLimits(1,0,250)

    # fabscor->SetParameter(0,0.0221278);
    # fabscor->SetParLimits(0,-2,50);
    # fabscor->SetParameter(1,14.265);
    # fabscor->SetParLimits(1,0,250);
    # fabscor->SetParameter(2,10);
    # fabscor->SetParLimits(2,0,200);
    # fabscor->SetParameter(3,-0.0679365);
    # fabscor->SetParLimits(3,-15,15);
    # fabscor->SetParameter(4,2.82597);
    # fabscor->SetParLimits(4,0,5);
    # fabscor->SetParameter(5,1.8277);
    # fabscor->SetParLimits(5,0,50);
    # fabscor->SetParameter(6,-0.0679365);
    # fabscor->SetParLimits(6,-20,10);
    # fabscor->SetParameter(7,3.82597);
    # fabscor->SetParLimits(7,0,100);
    # fabscor->SetParameter(8,1.8277);
    # fabscor->SetParLimits(8,-50,50);

    # func_root.SetParameters(
    #     0.0221278,
    #     14.265,
    #     10,
    #     -0.0679365,
    #     2.82597,
    #     1.8277,
    #     -0.0679365,
    #     3.82597,
    #     1.8277,
    # )
    func_root.SetParameters(
        0.4,
        16.265,
        11,
        -0.48,
        5,
        0.8277,
        # -0.54,
        # 0.249482,
        # 0.9277,
    )
    # func_root.SetParLimits(0, -2, 50)
    # func_root.SetParLimits(1, 0, 250)
    # func_root.SetParLimits(2, 0, 200)
    # func_root.SetParLimits(3, -15, 15)
    # func_root.SetParLimits(4, 0, 10)
    # func_root.SetParLimits(5, 0, 50)
    # func_root.SetParLimits(6, -20, 10)
    # func_root.SetParLimits(7, 0, 100)
    # func_root.SetParLimits(8, -50, 50)

    # ROOT.TFitter.SetPrecision(10.)

    graph = ROOT.TGraphErrors(len(x), x, y, xerr, yerr)

    fit_info = str(graph.Fit("func_root", "S N Q R EX0"))  # M E

    # "EX0" 	When fitting a TGraphErrors or TGraphAsymErrors do not consider errors in the X coordinates
    # “E” Perform better errors estimation using the Minos technique
    # “M” Improve fit results, by using the IMPROVE algorithm of TMinuit. (problematic?)

    num_pars = func_root.GetNpar()
    param_fit = [func_root.GetParameter(i) for i in range(num_pars)] + [0] * (
        9 - num_pars
    )
    param_err_fit = [func_root.GetParError(i) for i in range(num_pars)] + [0] * (
        9 - num_pars
    )

    # delete graph
    del graph

    # plot the fit
    x_fit = np.linspace(x[0], x[-1], 2000)
    y_fit = std_gaus(x_fit, *param_fit)

    # print chi2 and p-value on the plot
    # chi2 = np.sum(((y - std_gaus(x, *param_fit)) / yerr) ** 2)
    derivatives = np.array([func_root.Derivative(x[i]) for i in range(len(x))])
    chi2 = np.sum(
        (y - std_gaus(x, *param_fit)) ** 2 / (yerr**2 + (xerr * derivatives) ** 2)
    )
    ndof = len(x) - num_pars
    p_value = 1 - stats.chi2.cdf(chi2, ndof)

    # ax.plot(x_fit, y_fit, color=variables_plot_settings[variable][0], linestyle="--")
    # ax.text(
    #     0.98,
    #     0.2 + y_pos,
    #     f"$\chi^2$ / ndof = {chi2:.2f} / {ndof}" + f", p-value = {p_value:.2f}",
    #     horizontalalignment="right",
    #     verticalalignment="top",
    #     transform=ax.transAxes,
    #     color=variables_plot_settings[variable][0],
    #
    # )

    if True:  # "Invalid FitResult" not in str(fit_info):
        print(
            "\n",
            name_plot,
            "\nx",
            x,
            "\ny",
            y,
            "\nx_err",
            xerr,
            "\nyerr",
            yerr,
            "\npars",
            param_fit,
            "\nerr_pars",
            param_err_fit,
            "\nchi2/ndof",
            chi2,
            "/",
            ndof,
            "p_value",
            p_value,
            "\nfit_info",
            fit_info,
        )

    #     global VALID_FIT
    #     VALID_FIT += 1

    # if p_value > 0.05:
    #     global GOOD_FIT
    #     GOOD_FIT += 1
    # else:
    #     global BAD_FIT
    #     BAD_FIT += 1

    # global TOTAL_FIT
    # TOTAL_FIT += 1

    fit_results = {
        "x": list(x),
        "y": list(y),
        "yerr": list(yerr),
        "parameters": param_fit,
        "errors": param_err_fit,
        "chi2": chi2,
        "ndof": ndof,
        "p_value": p_value,
        "fit_info": fit_info,
    }

    return fit_results



## HOW TO USE THE FUNCTIONS

# fit_results = fit_inv_median_root(
#     ax,
#     x,
#     y,
#     # xerr,
#     np.zeros(len(x)),
#     y_err,
#     variable,
#     y_pos,
#     f"{eta_sign} {flav} {correct_eta_bins[eta_bin]} ({index}) {variable}",
# )