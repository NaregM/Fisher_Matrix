import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from scipy.integrate import quad, trapz, simps
from scipy.stats import chisquare
from scipy.special import erfc

import camb
from camb import model, initialpower

import scipy
from scipy.optimize import curve_fit

import copy

from multiprocessing import Pool
from Calibration import Calibration
from Distance import Distance
from Cosmos import Cosmos

# ===================================================

def dlnPh_dp(m, n, l, p, params, kmin, kmax, N_k, bfNL, y200 = 2e-3, dp_ = 2e-2):

    """
    Derivative of P_h^{m, n}

    """

    if p in ["fnl", "BM0", "alpha", "sLnM0", "beta"]:

        dp = 1e-2

    else:

        dp = params[p] * dp_

    p_p = params.copy()        #copy.copy(params)
    p_m = params.copy()        #copy.copy(params)

    p_p.update({p: params[p] + dp})
    p_m.update({p: params[p] - dp})


    c_p = Cosmos(l, kmin = kmin, kmax = kmax, N = N_k, Y200rhoc = y200, **p_p, bfNL_Ph = bfNL)
    c_p.setPower()
    c_p.setMlim()

    c_m = Cosmos(l, kmin = kmin, kmax = kmax, N = N_k, Y200rhoc = y200, **p_m, bfNL_Ph = bfNL)
    c_m.setPower()
    c_m.setMlim()

    P_h_p = c_p.Ph_mn(m, n)
    P_h_m = c_m.Ph_mn(m, n)

    return (np.log(P_h_p) - np.log(P_h_m))/(2*dp)


def F_ij(m, n, pa, pb, bfNL, l, y200, params, kmin = 1e-4, kmax = 6.0, N_k = 475):

    """
    Eq. 20 from 1003.0841
    multiprocessing version
    """
    c_fid = Cosmos(l, kmin = kmin, kmax = kmax, N = N_k, Y200rhoc = y200, **params, bfNL = False)
    c_fid.setPower()
    c_fid.setMlim()
    kd = c_fid.kd()
    Veff = c_fid.Veff_mn(m, n)

    lnPha = dlnPh_dp(m, n, l, pa, kmin = kmin, kmax = kmax, N_k = N_k, y200 = y200, params = params, bfNL = bfNL)
    lnPhb = dlnPh_dp(m, n, l, pb, kmin = kmin, kmax = kmax, N_k = N_k, y200 = y200, params = params, bfNL = bfNL)


    if m == n:

        return np.sum(lnPha * lnPhb * Veff * kd**2 * (0.017))

    elif m != n:

        return 2.0 * np.sum(lnPha * lnPhb * Veff * kd**2 * (0.017))


def Fij_mlp(p0, p1, bfnl, mn, zmax, y200, params):

    p_mlp = [[m[0], m[1], p[0], p[1], bfnl_, l, y200_, params_]
               for p, bfnl_ in zip([(p0, p1)], [bfnl])
               for m in mn
               for l in np.arange(0, zmax + 0.05, 0.05)
               for y200_ in [y200]
	       for params_ in [params]]

    pool = Pool()

    F_N = pool.starmap(F_ij, p_mlp)

    pool.close()

    return 1/(2 * np.pi)**2 * np.sum(F_N)
