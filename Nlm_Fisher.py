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

# =============================================================

def N_lm_v2(l, m, params, fsky = 0.7, y200 = 2e-3,
            zmin = 0.0, Delta_z = 0.05, kmin = 1e-4, kmax = 5.0, N = 600, bfNL = False, disable_pbar = True):
    
    """
    Eq. 14 from 1003.0841
    Approximate integral
    
    """

    z_l = zmin + l * Delta_z
    z_lp1 = zmin + (l+1) * Delta_z
    
    z_mid = 0.5 * (z_l + z_lp1)
    

    y200rhoc = y200  # arcmin^2
        
    c = Cosmos(z_mid, kmin, kmax, N, y200rhoc, bfNL = bfNL, **params)

    c.setPower()
    c.setMlim()
        
    Mlim = c.Mlim
                                    
    nM = np.array([c.dndM(M_) for M_ in c.Md()])
    nM = np.where(np.isnan(nM) == True, np.nanmin(nM) * 1e3*np.random.normal(), nM) 

    ERF = np.array([erfc(c.xm(M_, Mlim, m)) - erfc(c.xm(M_, Mlim, m + 1)) for M_ in c.Md()])
        
    V =  c.dVdz()
    
    N_ = trapz(nM * ERF * V, c.Md())
        
    

    I = fsky/2 *  N_ * Delta_z
            
    return I


def dNlm_dp(l, m, p, params, bfNL, y200, dp_ = 1e-2):
    
    """
    Derivative of N_lm
    
    """
    dp = params[p] * dp_
    
    p_p = copy.copy(params)
    p_m = copy.copy(params)
    
    p_p.update({p: params[p] + dp})
    p_m.update({p: params[p] - dp})

    Nlm_p = N_lm_v2(l, m, params = p_p, y200 = y200, bfNL = bfNL)
    Nlm_m = N_lm_v2(l, m, params = p_m, y200 = y200, bfNL = bfNL)
    
    return (Nlm_p - Nlm_m)/(2*dp)


def Fab(pa, pb, l, m, bfNL, y200_, params, dp_ = 1e-2):
    
    """
    Eq. 13 from 1003.0841
    """

    Nlm_fid = N_lm_v2(l, m, params = params, bfNL = False)
    dNlm_dpa = dNlm_dp(l, m, p = pa, y200 = y200_, params = params, dp_ = dp_, bfNL = bfNL)
    dNlm_dpb = dNlm_dp(l, m, p = pb, y200 = y200_, params = params, dp_ = dp_, bfNL = bfNL)

            
    return dNlm_dpa * dNlm_dpb * 1/Nlm_fid


def Fab_mlp(p0, p1, bfnl, lmax, mmax, y200, params):

    p_mlp = [[p[0], p[1], m, l, bfnl_, y200_, params_]
               for p, bfnl_ in zip([(p0, p1)], [bfnl])
               for m in range(1, mmax)
               for l in range(0, lmax)
               for y200_ in [y200]
	       for params_ in [params]]

    pool = Pool()

    F_N = pool.starmap(Fab, p_mlp)

    pool.close()

    return np.sum(F_N)

