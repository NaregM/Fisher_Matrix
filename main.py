import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import pickle

import itertools

from Calibration import Calibration
from Distance import Distance
from Cosmos import Cosmos

from Ph_Fisher import Fij_mlp
from Nlm_Fisher import Fab_mlp, Fab



# Fiducial parameters =====================================

p_fid = {'H0': 73.000000000,
        'Omegab0': 0.045974855,
         'Omegam0': 0.268343029,
         'ns': 0.963000000,
         'w0': -1.000000000,
         'sigma_8': 0.809000000}


p_fid_fnl = {'H0': 73.000000000,
             'Omegab0': 0.045974855,
             'Omegam0': 0.268343029,
             'ns': 0.963000000,
             'w0': -1.000000000,
             'sigma_8': 0.809000000,
             'fnl': 0.00}


# All possible unique pair of parameters. e.g. ("H0", "sigma8")
p_cross = list(itertools.combinations_with_replacement(list(p_fid.keys()), 2))

# ==========================================================

doNumberCounts = True
doPowerSpectrum = False

# 2e-3
if doNumberCounts:

    Y200c = 2.0e-3
    Lmax = 20
    Mmax = 7

    F_param = {}

    for i, p in enumerate(p_cross):

        bfnl_ = False

        _F = Fab_mlp(p[0], p[1], bfnl_, Lmax, Mmax, Y200c, p_fid)

        print(p, " -->> ", _F, ",", "\n")

        F_param.update({p:  _F})


if doPowerSpectrum:

    m_max = 7
    mn = list(itertools.combinations_with_replacement(range(m_max), 2))


    Y200c = 1.0e-3
    z_max = 1.0

    F_param = {}

    for i, p in enumerate(p_cross):

        if p[0] == "fnl" or p[1] == "fnl":

            #bfnl_ = True

            pass  #_F = Fij_mlp(p[0], p[1], bfnl_)

        else:

            bfnl_ = False

            _F = Fij_mlp(p[0], p[1], bfnl_, mn, z_max, Y200c, p_fid)

        print(p, " -->> ", _F, ",", "\n")

        F_param.update({p:  _F})



save_file = open("F_" + str(Y200c) + ".pkl", "wb")
pickle.dump(F_param, save_file)
save_file.close()
